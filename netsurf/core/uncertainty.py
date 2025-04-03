from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Set

import numpy as np
import scipy.stats
from scipy.stats import entropy as kl_divergence
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd

import seaborn as sns
 

import netsurf # For utils

# import pergamos
import pergamos as pg

DEFAULT_DISTRIBUTION_COLORS = {
    "fisher": "#FF8C42",       # Orange (strong highlight, conveys sensitivity)
    "aleatoric": "#4BA3C7",    # Sky Blue (clean, associated with variability)
    "msb": "#A463F2",          # Purple (represents discrete/bitwise influence)
    "msb_impact_ratio": "#A463F2",  # Purple (represents discrete/bitwise influence)
    "delta": "#3FC380",        # Green (change/gradient – intuitive for deltas)
    "qpolar": "#FF5C8A",       # Pinkish Red (emphasizes custom nature)
    "qpolargrad": "#D7263D",   # Deep Red (gradient + qpolar fusion = intense)
    "empirical_ber_accuracy": "#5B5F97",  # Indigo (robust/experimental nature)
    "ranking_effectiveness": "#F4A261",   # Muted Orange (signal performance)
}

DEFAULT_PLOT_FUNCTIONS = {
    'fisher': {'plot_distribution', 'plot_lorenz_curve', 'plot_perce_curve', 'plot_cumulative_mass'},
    'aleatoric': {'plot_distribution'},
    'msb': {'plot_distribution'},
    'delta': {'plot_distribution'},
    'qpolar': {'plot_distribution', 'plot_lorenz_curve', 'plot_perce_curve', 'plot_cumulative_mass'},
    'qpolargrad': {'plot_distribution', 'plot_lorenz_curve', 'plot_perce_curve', 'plot_cumulative_mass'},
    'empirical_ber_accuracy': {'plot_distribution'},
}



# +============================================================================================+
# | SOURCES OF UNCERTAINTY — QUANTIFICATION TIMING AND ROLE                                    |
# |--------------------------------------------------------------------------------------------|
# |    This table documents all types of uncertainty tracked in this project, grouped          |
# |    by category. Each block explains the nature of the uncertainty, the metric used         |
# |    to quantify it, and whether it is computed before or after bit-flip injection.          |
# |                                                                                            |
# +============================================================================================+
#
#
# +============================================================================================+
# | EPISTEMIC UNCERTAINTY                                                                      |
# |--------------------------------------------------------------------------------------------|
# |  - Captures the model's sensitivity to changes in its parameters (curvature).              |
# |  - Related to how well the model has learned / how sharp the minimum is.                   |
# |  - Used to quantify overall robustness of the learned parameters.                          |
# |---+------------------------------------+-----------------------------+---------------------|
# |   | Source                             | Metric / Signature          | When Computed       |
# |   |------------------------------------+-----------------------------+---------------------|
# |   | Fisher Information (curvature)     | Fisher (diag)               | Before BER          |
# |   | Fisher spectral shape              | DistributionSignature       | Before BER          |
# |---+------------------------------------+-----------------------------+---------------------|
# | ALEATORIC UNCERTAINTY                                                                      |
# |--------------------------------------------------------------------------------------------|
# |  - Captures input-driven uncertainty (e.g. noise, variability in input features).          |
# |  - Measured by gradient of loss w.r.t inputs.                                              |
# |  - Used to detect instability to input shifts, e.g. natural corruption or sampling noise.  |
# |---+------------------------------------+-----------------------------+---------------------|
# |   | Source                             | Metric / Signature          | When Computed       |
# |   |------------------------------------|-----------------------------|---------------------|
# |   | Input sensitivity                  | ∇_X L norm stats            | Before BER          |
# |---+------------------------------------+-----------------------------+---------------------|
# | STRUCTURAL UNCERTAINTY                                                                     |
# |--------------------------------------------------------------------------------------------|
# |  - Captures sensitivity arising from the binary representation and weight topology.        |
# |  - Related to quantization, bit-level deltas, and weight shape.                            |
# |  - Used to estimate model susceptibility to bit-level corruption.                          |
# |---+------------------------------------+-----------------------------+---------------------|
# |   | Source                             | Metric / Signature          | When Computed       |
# |   |------------------------------------+-----------------------------+---------------------|
# |   | QPolar impact (bit energy)         | QPolar, QPolarGrad          | Before BER          |
# |   | Bit impact distribution shape      | DistributionSignature       | Before BER          |
# |   | Delta spectrum skew/kurtosis       | DistributionSignature       | Before BER          |
# |   | Bit importance (MSB)               | MSB impact ratio            | Before BER          |
# +============================================================================================+
#
#
# +============================================================================================+
# | EMPIRICAL / OBSERVED EFFECTS (AFTER BER)                                                   |
# |--------------------------------------------------------------------------------------------|
# |  - Captures real-world degradation after fault injection.                                  |
# |  - Used to measure ranking effectiveness and model resilience under perturbation.          |
# |  - Used to estimate model susceptibility to bit-level corruption.                          |
# |---+------------------------------------+-----------------------------+---------------------|
# |   | Source                             | Metric / Signature          | When Computed       |
# |   |------------------------------------|-----------------------------|---------------------|
# |   | Bit-flip degradation               | Accuracy @ BER              | After BER           |
# |   | Ranking effectiveness              | Acc vs protected bits curve | After BER           |
# +============================================================================================+


@dataclass
class DistributionSignature:
    _ICON = "🖋️"
    # Class name for the signature (e.g., Fisher, QPolar, etc.)
    name: Optional[str] = "DistributionSignature"
    # Actual data 
    data: Optional[np.ndarray] = None
    # Shannon entropy of the normalized distribution (e.g., bit impact or delta)
    entropy: Optional[float] = None
    # Variance of the values in the distribution
    variance: Optional[float] = None
    # Skewness (asymmetry) of the distribution
    skewness: Optional[float] = None
    # Kurtosis (peakedness/tail weight) of the distribution
    kurtosis: Optional[float] = None
    # Mean value of the distribution
    mean: Optional[float] = None
    # Standard deviation of the distribution
    std: Optional[float] = None
    # Minimum value in the distribution
    min: Optional[float] = None
    # Maximum value in the distribution
    max: Optional[float] = None
    # L1 norm of the distribution (sum of absolute values)
    l1_energy: Optional[float] = None
    # L2 norm of the distribution (sum of squared values)
    l2_energy: Optional[float] = None
    # Gini coefficient (0 = perfect equality, 1 = perfect inequality)
    gini: Optional[float] = None
    # Color for the plot (default: None)
    color: Optional[str] = 'skyblue'
    # We can also define the bins 
    # (e.g., for deltas these can only be very specific values due to the quantization)
    bins: Optional[np.ndarray] = None

    # This is true for deltas
    quantized: Optional[bool] = False  # New field
    is_data_pdf: Optional[bool] = False  # New field

    # Additional fields for specific distributions
    # (e.g., for fisher this will hold the trace (diag/square), etc.)
    _specific_fields: Optional[dict] = None  # New field
    _plotting_methods: set = field(default_factory=lambda: {'plot_distribution'})

    @staticmethod
    def compute_gini(x):
        x = np.abs(np.sort(x))
        n = len(x)
        if n == 0:
            return None
        index = np.arange(1, n + 1)
        return (np.sum((2 * index - n - 1) * x)) / (n * np.sum(x) + 1e-8)

    @staticmethod
    def from_array(array: np.ndarray, name=None, extra_fields: Optional[dict] = None, color = None, bins = None, 
                   quantized = False, is_data_pdf = False, plot_functions = {'plot_distribution'}) -> "DistributionSignature":
        # Flattens input array, computes statistics, returns DistributionSignature
        
        flat = array.flatten()

        if is_data_pdf:
            # We can consider the data as a pdf directly (for instance for fisher)
            p = np.abs(flat) / (np.sum(np.abs(flat)) + 1e-8)  # Normalize for entropy
            entropy = -np.sum(p * np.log(p + 1e-8))
        else:
            # We need to compute the histogram
            # Use Freedman-Diaconis rule to determine the number of bins
            num_bins = netsurf.utils.math.freedman_diaconis_bins(flat)
            # Clip by a maximum of 256 bins
            _bins = min(num_bins, 256)
            if _bins < num_bins:
                print(f'Freedman-Diaconis rule suggested {num_bins} bins, but using {_bins} instead to avoid memory issues.')
            p, bin_edges = np.histogram(flat, bins = _bins, density=True)
            bin_widths = np.diff(bin_edges)
            entropy = -np.sum(p * np.log(p + 1e-8) * bin_widths)


        # Compute gini coefficient
        gini = DistributionSignature.compute_gini(flat)
        
        sig = DistributionSignature(
            data=flat,
            name=name,
            entropy=entropy,
            variance=np.var(flat),
            skewness=scipy.stats.skew(flat),
            kurtosis=scipy.stats.kurtosis(flat),
            mean=np.mean(flat),
            std=np.std(flat),
            min=np.min(flat),
            max=np.max(flat),
            l1_energy=np.sum(np.abs(flat)),
            l2_energy=np.sum(flat**2),
            gini=gini,
            color=color if color else DEFAULT_DISTRIBUTION_COLORS.get(name, 'skyblue'),
            bins=bins,
            quantized = quantized,
            _plotting_methods = plot_functions
        )

        # Store and assign method-specific fields
        sig._specific_fields = extra_fields
        if extra_fields:
            for k, v in extra_fields.items():
                setattr(sig, k, v)
        
        return sig
    
    @property
    def count(self) -> int:
        """
        Returns the number of elements in the distribution.
        """
        return len(self.data) if self.data is not None else 0

    def plot_distribution(self, bins: int = None, figsize: Tuple[int, int] = (8, 7), 
         title: str = None, logbin: bool = True, ax=None, show = True, color = None,
         log2 = False):
        """
        Plot the histogram of the data with summary statistics in a boxed annotation.

        Parameters:
            bins (int): Number of histogram bins.
            figsize (tuple): Size of the matplotlib figure.
            title (str): Optional plot title.
            logbin (bool): Whether to apply log10 binning if data spans multiple orders of magnitude.
            ax: Optional matplotlib axis to draw on.
        """
        flat = self.data
        show &= ax is None 
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        log2 = log2 or self.quantized
        
        # Use log10 binning if enabled and necessary
        use_logbin = False
        if logbin and not log2:
            positive = flat[flat > 0]
            if len(positive) > 0:
                ratio = np.max(positive) / np.min(positive)
                if ratio > 1e3:
                    use_logbin = True
                    flat = np.log10(positive)
                    ax.set_xlabel(r"$log_{10}$(Value)")
        
        if not use_logbin:
            ax.set_xlabel("Value")

        color = color if color else self.color

        # Check bins
        if bins is None:
            if self.bins is not None:
                bins = self.bins
            else:
                # Default number of bins
                bins = 50


        if not log2:
            # # Just use default pyplot method
            # ax.hist(flat, bins=bins, color=color, edgecolor='k', alpha=0.7, density=True)

            # Compute and plot the CDF on a secondary Y-axis
            hist, bin_edges = np.histogram(flat, bins=bins, density=True)
            cdf = np.cumsum(hist * np.diff(bin_edges))
            cdf = cdf / cdf[-1]  # Normalize to 1

            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            ax.bar(bin_centers, hist, width=np.diff(bin_edges), color=color, edgecolor='k', alpha=0.7, label = 'PDF')

            # Twin axis for CDF
            ax2 = ax.twinx()
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            ax2.plot(bin_centers, cdf, color='tab:red', linewidth=1.5, linestyle='-', label='CDF')
            ax2.set_ylabel("Cumulative Probability")
            ax2.tick_params(axis='y', labelcolor='tab:red')

            # Add KDE line if enough data points
            if len(flat) > 1:
                kde = scipy.stats.gaussian_kde(flat)
                x_kde = np.linspace(np.min(flat), np.max(flat), 500)
                ax.plot(x_kde, kde(x_kde), color='black', linestyle='--', linewidth=1.5, label='KDE')
            
            # Optionally add legend
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            # Combine legends
            ax.legend(lines + lines2, labels + labels2, loc = 'upper left')


        else:
            # This is trickier. Most likely the bins are a power of 2. Meaning, if
            # we just plot them using hist, the bin widths will have different sizes.
            # So we need to plot them manually
            # Get the bin edges manually
            counts = [np.sum(flat == bins[i]) for i in range(len(bins))]
            _bins = np.arange(-len(bins)//2 + 1, len(bins)//2 + 1)
            plt.rcParams['text.usetex'] = True
            ax.bar(_bins, counts, width=1, color=color, edgecolor='k', alpha=0.7)

            # Set the labels 
            ax.set_xticks(_bins)
            def format(v):
                if v == 0:
                    return r"$0$"
                elif v > 0:
                    if v >= 1:
                        # Use value itself
                        return rf"${int(v)}$"
                    elif v < 1:
                        return r'$-2^{' + f'{int(np.log2(v))}' + r'}$'
                else:
                    if v <= -1:
                        # Use value itself
                        return rf"$-{-int(v)}$"
                    elif v > -1:
                        return r'$-2^{' + f"{int(np.log2(-v))}" + r'}$'
                

            nticks = 10
            # max num of ticks 
            if len(bins) > nticks:
                # Get the tick positions
                ticks = bins[::len(bins)//nticks]
                ticks_pos = _bins[::len(bins)//nticks]
                # Get the tick labels
                labels = [format(x) for x in ticks]
                # Set the ticks and labels
                ax.set_xticks(ticks_pos)
                ax.set_xticklabels(labels, rotation=0)
            else:
                ax.set_xticklabels([format(x) for x in bins], rotation=0)
                ax.set_xticks(bins, minor=True)

            ax.set_xlabel(r"Value ($log_2$ scale)")

        ax.set_ylabel("Frequency")

        # Add summary box (at the top of the plot)
        num_columns = 2
        def format_value(val):
            if val is None:
                return "N/A"
            else:
                return f"{val:.4f}" if np.log(np.abs(val) + 1e-8) < 3 else f"{val:.2e}"
        
        # Init stats with empty string
        stats = ['','']
        # Add specific fields if any
        if self._specific_fields:
            for k, v in self._specific_fields.items():
                stats.append('')
                stats.append(f"${k}$: {format_value(v)}")
        
        stats += [
            rf"$\mathcal{{H}}$ (entropy): {format_value(self.entropy)}",
            rf"$\sigma^2$ (variance): {format_value(self.variance)}",
            rf"$\gamma_1$ (skew): {format_value(self.skewness)}",
            rf"$\gamma_2$ (kurtosis): {format_value(self.kurtosis)}",
            rf"$\mu$ (mean): {format_value(self.mean)}",
            rf"$\sigma$ (std): {format_value(self.std)}",
            rf"$min$: {format_value(self.min)}",
            rf"$max$: {format_value(self.max)}",
            rf"$\|x\|_1$ (L1 energy): {format_value(self.l1_energy)}",
            rf"$\|x\|_2^2$ (L2 energy): {format_value(self.l2_energy)}",
            rf"$\mathcal{{G}}$ (gini): {format_value(self.gini)}",
        ]

        # Reshape into columns
        stats = [" || ".join(stats[i:i + num_columns]) if all([s != '' for s in stats[i:i + num_columns]]) else "".join(stats[i:i + num_columns]) \
                 for i in range(0, len(stats), num_columns)]

        stats.insert(0, r"\textbf{\Large Distribution for " + self.name.capitalize() + r" }" if not title else r'\textbf{\Large ' + title.capitalize() + r" }")
        textstr = '\n'.join(stats)
        props = dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.95)
        ax.text(0.5, 1.05, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='bottom', horizontalalignment='center', 
                bbox=props, usetex=True)

        # Turn axis 
        netsurf.utils.plot.turn_grids_on(ax)
        
        if show: 
            plt.tight_layout()
            plt.show()
            plt.rcParams['text.usetex'] = False

        return fig, ax
    
    
    ## Lorentz curve
    def plot_lorenz_curve(self, figsize=(8, 6), title=None, ax=None, show = True, **kwargs):
        """
        Plots the Lorenz (Pareto) curve of the distribution.

        The Lorenz curve shows the cumulative proportion of the total "mass" 
        (e.g., Fisher information, impact) accounted for by the bottom x% 
        of sorted elements. 

        - X-axis: Cumulative fraction of parameters (sorted by magnitude)
        - Y-axis: Cumulative contribution to total value

        A perfectly uniform distribution produces a diagonal line.
        A highly skewed distribution (e.g. a few dominant weights) 
        bends sharply below the diagonal.

        Useful for diagnosing inequality in parameter importance.

        Parameters:
            ax (matplotlib.Axes): Optional axis to plot on.
            figsize (tuple): Figure size if `ax` is not provided.
            show (bool): Whether to display the plot immediately.
        """
        show &= (ax is None)
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        # Sort data
        data = np.sort(np.abs(self.data))
        n = len(data)
        cum_data = np.cumsum(data)
        cum_data = cum_data / cum_data[-1]  # Normalize
        x = np.linspace(0.0, 1.0, n)

        # if we have the gini coefficient, we can add it to the label
        if self.gini is not None:
            title = title if title else f"Lorenz Curve for {self.name} (Gini: {self.gini:.4f})"

        # Plot Lorenz curve
        ax.plot(x, cum_data, label="Lorenz Curve", color=self.color)
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Equality")
        ax.set_title(title if title else f"Lorenz Curve for {self.name}")
        ax.set_xlabel("Cumulative Fraction of Parameters")
        ax.set_ylabel("Cumulative Fisher Mass")
        ax.legend()
        netsurf.utils.plot.turn_grids_on(ax)
        if show:
            plt.tight_layout()
            plt.show()
        return fig, ax

    # Perce curve
    def plot_perce_curve(self, figsize=(8, 6), title=None, ax=None, show = True, **kwargs):
        """
        Plots the cumulative mass curve of the distribution.

        Similar to the Lorenz curve, but with a linear X-axis.

        - X-axis: Sorted parameter index (0 to N)
        - Y-axis: Cumulative sum of the absolute values

        This plot shows how quickly the total "energy" (e.g., Fisher mass)
        accumulates across parameters. Useful for understanding how many
        parameters carry the bulk of the signal.

        Parameters:
            ax (matplotlib.Axes): Optional axis to plot on.
            figsize (tuple): Figure size if `ax` is not provided.
            show (bool): Whether to display the plot immediately.
        """
        show &= (ax is None)
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        data = np.sort(np.abs(self.data))
        n = len(data)
        cum_data = np.cumsum(data)
        cum_data = cum_data / cum_data[-1]  # Normalize
        x = np.linspace(0.0, 1.0, n)

        midpoint = np.searchsorted(cum_data, 0.5) / n

        # Recenter
        x_shifted = x - midpoint

        ax.plot(x_shifted, cum_data, label="Perce Curve", color=self.color)
        ax.axhline(0.5, color="gray", linestyle="--")
        ax.axvline(0.0, color="gray", linestyle=":")
        ax.set_title(title if title else f"Perce Curve for {self.name}")
        ax.set_xlabel("Centered Rank (w.r.t 50% mass)")
        ax.set_ylabel("Cumulative Fisher Mass")
        ax.legend()
        netsurf.utils.plot.turn_grids_on(ax)
        if show:
            plt.tight_layout()
            plt.show()
        return fig, ax

    # Cumulative mass
    def plot_cumulative_mass(self, figsize=(8, 6), title=None, ax=None, show = True, **kwargs):
        """
        Plots the Perce curve of the distribution.

        The Perce curve re-centers the Lorenz curve at the 50% mark,
        making it easier to compare how much mass lies above/below 
        the expected uniform contribution.

        - X-axis: Centered around 0 (i.e., [-0.5, 0.5])
        - Y-axis: Cumulative mass deviation from uniform

        Useful for highlighting divergence from even distributions 
        and visualizing saturation vs sparsity effects.

        Parameters:
            ax (matplotlib.Axes): Optional axis to plot on.
            figsize (tuple): Figure size if `ax` is not provided.
            show (bool): Whether to display the plot immediately.
        """
        show &= (ax is None)
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        data = np.sort(np.abs(self.data))[::-1]
        cumulative = np.cumsum(data) / np.sum(data)
        ax.plot(range(len(cumulative)), cumulative, color=self.color, label="Cumulative Mass")
        ax.set_title(title if title else f"Cumulative Mass for {self.name}")
        ax.set_xlabel("Parameter Rank (sorted by importance)")
        ax.set_ylabel("Cumulative Mass")
        ax.legend()
        netsurf.utils.plot.turn_grids_on(ax)
        if show:
            plt.tight_layout()
            plt.show()
        return fig, ax

    def __repr__(self):
        tab = ' '*4
        # Custom string representation for DistributionSignature
        s = f'{self._ICON} {self.name.capitalize()} {self.__class__.__name__}\n'
        # First add the specific fields
        if self._specific_fields:
            for k, v in self._specific_fields.items():
                s += f"{tab}⭐️ {k}={v}\n"
        s += f"{tab}🌀 \U0000210B (entropy)={self.entropy:.4f}\n"
        s += f"{tab}📏 \u03C3\u00B2 (variance)={self.variance:.4f}\n"
        s += f"{tab}🦂 \u03B3\u2081 (skewness)={self.skewness:.4f}\n"
        s += f"{tab}🏔️ \u03B3\u2082 (kurtosis)={self.kurtosis:.4f}\n"
        s += f"{tab}🎯 \u03BC (mean)={self.mean:.4f}\n"
        s += f"{tab}⭕️ \u03C3 (std)={self.std:.4f}\n"
        s += f"{tab}🔽 min={self.min:.4f}\n"
        s += f"{tab}🔼 max={self.max:.4f}\n"
        s += f"{tab}⚡️ \u2016x\u2016\u2081 (l1_energy)={self.l1_energy:.4f}\n"
        s += f"{tab}💪 \u2016x\u2016\u2082 (l2_energy)={self.l2_energy:.4f}\n"
        return s
    
    def html(self):
        # Create collapsible container for this signature
        sig_ct = pg.CollapsibleContainer(f"{self._ICON} {self.name}", layout='vertical')
        # Add the stats as a non-collapsible container table 
        # Create a container showing the basic information summary for this summary 
        summary_ct = pg.Container("Summary", layout='vertical')

        def format_value(val):
            if val is None:
                return "N/A"
            else:
                return f"{val:.4f}" if np.log(np.abs(val) + 1e-8) < 3 else f"{val:.2e}"
        df = {}
        if self._specific_fields:
            for k, v in self._specific_fields.items():
                df[f"${k}$"] = format_value(v)
        
        df[r"<div class='math-content'>$\mathcal{H}$</div> (entropy)"] = format_value(self.entropy)
        df[r"<div class='math-content'>$\sigma^2$</div> (variance)"] = format_value(self.variance)
        df[r"<div class='math-content'>$\gamma_1$</div> (skew)"] = format_value(self.skewness)
        df[r"<div class='math-content'>$\gamma_2$</div> (kurtosis)"] = format_value(self.kurtosis)
        df[r"<div class='math-content'>$\mu$</div> (mean)"] = format_value(self.mean)
        df[r"<div class='math-content'>$\sigma$</div> (std)"] = format_value(self.std)
        df[r"<div class='math-content'>$min$</div>"] = format_value(self.min)
        df[r"<div class='math-content'>$max$</div>"] = format_value(self.max)
        df[r"<div class='math-content'>$\|x\|_1$</div> (L1 energy)"] = format_value(self.l1_energy)
        df[r"<div class='math-content'>$\|x\|_2^2$</div> (L2 energy)"] = format_value(self.l2_energy)
        df[r"<div class='math-content'>$\mathcal{G}$</div> (gini)"] = format_value(self.gini)


        # Add the specific fields to the summary
        df = dict(**{kv: getattr(self, kv) for kv in ['name', 'count']}, **df)
        
        # Create pandas dataframe 
        df = pd.DataFrame([df]).T
        
        # Add to container
        summary_ct.append(pg.Table.from_data(df))
        
        # Add summary container to session container
        sig_ct.append(summary_ct)

        # Container for the distribution plot 
        for method in self._plotting_methods:
            # Check if the method exists
            if hasattr(self, method):
                fcn = getattr(self, method)
                # Create a container for the plot
                plot_ct = pg.CollapsibleContainer(f"📊 {method.capitalize()}", layout='vertical')
                # Check if there is documentation for this method
                if method in netsurf.doc.uncertainty.UNCERTAINTY_PLOTS_DOC:
                    doc = netsurf.doc.uncertainty.UNCERTAINTY_PLOTS_DOC[method]
                    # Add the documentation to the container
                    plot_ct.append(pg.Markdown(doc, attributes={'style': 'padding: 10px;'}))
                

                # Add the plot to the container
                fig, ax = fcn(ax=None, show=False)
                # Convert the plot to an image
                img = pg.Image(fig, embed=True)
                # Add the image to the container
                plot_ct.append(img)
                sig_ct.append(plot_ct)
                # CLose fig
                plt.close(fig)


        # MAke sure to add mathjax to required_scripts
        sig_ct.required_scripts.add("mathjax")

        # Return the container
        return sig_ct



@dataclass
class RobustnessDivergence:
    method: str
    divergence_stats: Dict[str, float] = field(default_factory=dict)
    cosine_similarity: float = 0.0
    gini_shift: float = 0.0
    radar_shift: np.ndarray = field(default_factory=lambda: np.array([]))
    kl_divergence: float = 0.0
    distribution_stats: Dict[str, np.ndarray] = field(default_factory=dict)
    
    @staticmethod
    def from_signatures(pre: DistributionSignature, post: DistributionSignature) -> "RobustnessDivergence":
        """
        Computes divergence metrics between two DistributionSignatures.
        Useful to assess how training changes robustness structure.
        """
        stats_names = ["entropy", "variance", "skewness", "kurtosis", "mean", "std", "l1_energy", "l2_energy", "gini"]
        pre_vals = np.array([getattr(pre, k, 0.0) for k in stats_names])
        post_vals = np.array([getattr(post, k, 0.0) for k in stats_names])

        delta_stats = {k: post_v - pre_v for k, pre_v, post_v in zip(stats_names, pre_vals, post_vals)}

        cosine_sim = np.dot(pre_vals, post_vals) / (np.linalg.norm(pre_vals) * np.linalg.norm(post_vals) + 1e-8)
        gini_shift = (post.gini or 0.0) - (pre.gini or 0.0)
        radar_shift = post_vals - pre_vals
        # Compute symmetric KL divergence
        pre_pdf = np.abs(pre_vals) / (np.sum(np.abs(pre_vals)) + 1e-8)
        post_pdf = np.abs(post_vals) / (np.sum(np.abs(post_vals)) + 1e-8)
        kl1 = kl_divergence(pre_pdf + 1e-8, post_pdf + 1e-8)
        kl2 = kl_divergence(post_pdf + 1e-8, pre_pdf + 1e-8)
        kl_div = 0.5 * (kl1 + kl2)

        return RobustnessDivergence(
            method=post.name,
            divergence_stats=delta_stats,
            cosine_similarity=cosine_sim,
            gini_shift=gini_shift,
            radar_shift=radar_shift,
            distribution_stats={"pre": pre_vals, "post": post_vals},
            kl_divergence=kl_div
        )



@dataclass
class RobustnessSignature:
    _ICON = "🪖"
    # Epistemic Uncertainty (sensitivity to parameter changes)
    fisher_distribution: Optional[DistributionSignature] = None  # Full statistical profile of Fisher diagonals

    # Aleatoric Uncertainty (input gradient statistics)
    aleatoric_distribution: Optional[DistributionSignature] = None  # Norm stats of ∇_X L

    # Bit-level structural uncertainty
    msb_impact_ratio: Optional[DistributionSignature] = None  # Fraction of bit-level impact attributed to MSBs
    msb_impact_ratio_abs: Optional[DistributionSignature] = None  # Fraction of bit-level impact attributed to MSBs (absolute)
    msb_impact: Optional[DistributionSignature] = None  # Distribution of MSB impacts
    total_impact: Optional[DistributionSignature] = None  # Distribution of total impacts
    msb_impact_abs: Optional[DistributionSignature] = None  # Distribution of absolute MSB impacts
    total_impact_abs: Optional[DistributionSignature] = None  # Distribution of absolute total impacts

    # Structural uncertainty (bit-level deltas)
    delta_distribution: Optional[DistributionSignature] = None
    qpolar_distribution: Optional[DistributionSignature] = None
    qpolargrad_distribution: Optional[DistributionSignature] = None

    # Empirical/Observed metrics (computed after BER)
    empirical_ber_accuracy: Optional[DistributionSignature] = None  # Accuracy after bit flip
    ranking_effectiveness: Optional[DistributionSignature] = None  # Ranking impact on protection

    @property
    def fisher(self) -> DistributionSignature:
        """
        Returns the Fisher distribution signature.
        """
        return self.fisher_distribution
    
    @property
    def aleatoric(self) -> DistributionSignature:
        """
        Returns the aleatoric distribution signature.
        """
        return self.aleatoric_distribution
    
    @property
    def msb(self) -> DistributionSignature:
        """
        Returns the MSB impact ratio.
        """
        return self.msb_impact_ratio, self.msb_impact_ratio_abs, self.msb_impact, self.total_impact, self.msb_impact_abs, self.total_impact_abs
    

    @property
    def delta(self) -> DistributionSignature:
        """
        Returns the delta distribution signature.
        """
        return self.delta_distribution
    
    @property
    def qpolar(self) -> DistributionSignature:
        """
        Returns the QPolar distribution signature.
        """
        return self.qpolar_distribution
    
    @property
    def qpolargrad(self) -> DistributionSignature:
        """
        Returns the QPolarGrad distribution signature.
        """
        return self.qpolargrad_distribution
    
    @property
    def empirical(self) -> Tuple[float, float]:
        """
        Returns the empirical metrics (accuracy and ranking effectiveness).
        """
        return self.empirical_ber_accuracy, self.ranking_effectiveness
    
    """
        Plot all dists
    """
    def plot_distributions(self, methods = None, bins: int = None, figsize: Tuple[int, int] = (8, 7), logbin=True):
        if methods is None:
            methods = ["fisher", "aleatoric", "msb", "delta", "qpolar", "qpolargrad"]
        
        for i, method in enumerate(methods):
            # Get the method name and plot
            method = method.lower()
            if hasattr(self, method):
                sigs = getattr(self, method)
                if not isinstance(sigs, list) and not isinstance(sigs, tuple):
                    sigs = [sigs]
                # Plot each signature
                for sig in sigs:
                    if isinstance(sig, DistributionSignature):
                        # Plot the signature
                        fig, ax = sig.plot_distribution(bins=bins, figsize=figsize, logbin=logbin, log2 = sig.quantized or method=='delta')
                        ax.set_title(f"{method.capitalize()} Signature {i+1}")
                        plt.show()
                    else:
                        fig, ax = plt.subplots(figsize=figsize)
                        ax.set_title(f"{method.capitalize()} Signature {i+1}")
                        ax.text(0.5, 0.5, f"No data for {method}", ha='center', va='center')
            else:
                fig, ax = plt.subplots(figsize=figsize)
                ax.text(0.5, 0.5, f"Unknown method: {method}", ha='center', va='center')
    

    """ Plot pairplot between dists """
    def plot_correlations(self, methods=None, figsize: Tuple[int, int] = (8, 16), sample_size: int = -1, axs = None, show = True):
        """
        Plots a heatmap of correlation coefficients between the summary statistics of each DistributionSignature.
 
        Args:
            methods (list): List of methods to include. Default is all.
            figsize (tuple): Size of the matplotlib figure.
            sample_size (int): Max number of points to sample from each distribution to compute correlations.
        """
        if methods is None:
            methods = ["fisher", "aleatoric", 'msb_impact_ratio', "delta", "qpolar", "qpolargrad"]
 
        # if sample_size == -1, we need to find the minimum number of samples 
        if sample_size == -1:
            sample_size = min([getattr(self, method).count for method in methods if getattr(self, method) is not None])

        # Collect raw data into a DataFrame
        stats_data = {}
        stats_summary = {}
        for method in methods:
            sigs = getattr(self, method, None)
            if sigs is None:
                continue
            if not isinstance(sigs, list):
                sigs = [sigs]
            for sig in sigs:
                if isinstance(sig, DistributionSignature) and sig.data is not None:
                    data = sig.data
                    if (sample_size > 0) and (len(data) > sample_size):
                        data = np.random.choice(data, sample_size, replace=False)
                    stats_data[method] = data
 
                    # Also store summary statistics
                    stats_summary[method] = {
                        "entropy": sig.entropy,
                        "variance": sig.variance,
                        "skewness": sig.skewness,
                        "kurtosis": sig.kurtosis,
                        "mean": sig.mean,
                        "std": sig.std,
                        "min": sig.min,
                        "max": sig.max,
                        "l1_energy": sig.l1_energy,
                        "l2_energy": sig.l2_energy,
                        "gini": sig.gini,
                    }
        
        if len(stats_data) < 2:
            print("Not enough data to compute correlations.")
            return
 
        # Build data correlation matrix
        df_data = pd.DataFrame(dict(stats_data))
        corr_data = df_data.corr()
 
        # Build statistics correlation matrix
        df_stats = pd.DataFrame(stats_summary).T
        corr_stats = df_stats.corr()
 
        if axs is not None:
            # check size
            if not isinstance(axs, list) and not isinstance(axs, tuple) and not isinstance(axs, np.ndarray):
                axs = [axs]
            
            if len(axs) != 2:
                print("Invalid number of axes provided. Expected 2 axes. Creating new figure")
                axs = None

        # Plot both heatmaps
        show &= (axs is None)
        if axs is None:
            fig, axs = plt.subplots(2, 1, figsize=figsize)
        else:
            fig = axs[0].figure
            #fig.set_size_inches(figsize)
 
        sns.heatmap(corr_data, annot=True, fmt=".2f", cmap="coolwarm", square=True, linewidths=0.5, ax=axs[0])
        axs[0].set_title("Correlation of Raw Distributions")
 
        sns.heatmap(corr_stats, annot=True, fmt=".2f", cmap="coolwarm", square=True, linewidths=0.5, ax=axs[1])
        axs[1].set_title("Correlation of Summary Statistics")
 
        plt.tight_layout()
        if show: plt.show()

        return fig, axs


    def plot_radar_fingerprints(self, methods=None, figsize: Tuple[int, int] = (3, 3), axs = None, show = True):
        """
        Plots three radar (spider) charts per distribution method:
        - Min-max normalized stats
        - Z-score standardized stats
        - Log-scaled raw values
 
        Args:
            methods (list): List of method names (e.g., "fisher", "qpolar").
            figsize (tuple): Size of each radar figure.
            axs: Optional list of matplotlib axes.
            show (bool): Whether to display the plots immediately.
        """
        if methods is None:
            methods = ["fisher", "aleatoric", "msb_impact_ratio", "delta", "qpolar", "qpolargrad"]
 
        stats_names = ["entropy", "variance", "skewness", "kurtosis", "mean", "std", "l1_energy", "l2_energy", "gini"]
        stats_symbol = ["\U0000210B", "\u03C3\u00B2", "\u03B3\u2081", "\u03B3\u2082", "\u03BC", "\u03C3", "\u2016x\u2016\u2081", "\u2016x\u2016\u2082", "Gini"]
        all_values = {method: [] for method in methods}
        available_methods = []
 
        for method in methods:
            sig = getattr(self, method, None)
            if isinstance(sig, DistributionSignature):
                values = [getattr(sig, stat) if getattr(sig, stat) is not None else 0.0 for stat in stats_names]
                all_values[method] = values
                available_methods.append(method)
 
        if not available_methods:
            print("No valid methods found.")
            return
 
        df = pd.DataFrame(all_values, index=stats_names)
 
        df_minmax = (df - df.min(axis=1).values[:, None]) / (df.max(axis=1).values - df.min(axis=1).values + 1e-8)[:, None]
        df_z = (df - df.mean(axis=1).values[:, None]) / (df.std(axis=1).values + 1e-8)[:, None]
        df_log = np.log1p(df.clip(lower=1e-8))
 
        n = len(available_methods)
        fig, axs = plt.subplots(nrows=n, ncols=3, figsize=(figsize[0]*3*0.9, figsize[1]*n*1.05),
                                subplot_kw=dict(polar=True))
        column_titles = ['Min-Max Norm', 'Z-Score Norm', 'Log Scale']
        for j, title in enumerate(column_titles):
            fig.text(0.23 + j * 0.26, 0.95, title, ha='center', va='center', fontsize=10, weight='bold')
        plt.subplots_adjust(hspace=0.4)
 
        for i, method in enumerate(available_methods):
            for j, df_plot in enumerate([df_minmax, df_z, df_log]):
                values = df_plot[method].tolist()
                values += values[:1]
                angles = np.linspace(0, 2 * np.pi, len(stats_names), endpoint=False).tolist()
                angles += angles[:1]

                ax = axs[i, j] if n > 1 else axs[j]
                ax.plot(angles, values, label=method, linewidth=2, color=DEFAULT_DISTRIBUTION_COLORS[method])
                ax.fill(angles, values, alpha=0.2, color=DEFAULT_DISTRIBUTION_COLORS[method])
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(stats_symbol)
                ax.set_yticklabels([])
 
            # Draw rounded rectangle around the entire row
            row_axes = axs[i] if n > 1 else axs
            fig.canvas.draw()
            bbox_start = row_axes[0].get_position()
            bbox_end = row_axes[-1].get_position()
            x0 = bbox_start.x0 - 0.01
            y0 = bbox_end.y0 - 0.015
            width = bbox_end.x1 - bbox_start.x0 + 0.02
            height = bbox_start.y1 - bbox_end.y0 + 0.03
            rect = patches.FancyBboxPatch((x0, y0), width, height,
                                          boxstyle="round,pad=0.01", linewidth=1,
                                          edgecolor="black", facecolor="none", transform=fig.transFigure, zorder=10)
            fig.patches.append(rect)
            # Place the method title at the left edge of the row
            fig.text(x0 - 0.02, y0 + height / 2, method, va="center", ha="right", fontsize=10, weight='bold')
        if show:
            plt.tight_layout()
            plt.show()
 
        return fig, axs
        
    def html(self):
        # Create a pg object that can be rendered in HTML
        # Create collapsible container for this Signature
        obj_ct = pg.CollapsibleContainer(f"{self._ICON} {self.__class__.__name__}", layout='vertical')

        # Create a container for individual methods 
        methods_ct = pg.CollapsibleContainer("🎨 Methods", layout='vertical')
        # Add the methods container to the main container
        obj_ct.append(methods_ct)

        # Loop thru all methods and append their html
        for method in ['fisher', 'aleatoric', 'msb', 'delta', 'qpolar', 'qpolargrad']:
            # Get the method name and html
            method = method.lower()
            if hasattr(self, method):
                sigs = getattr(self, method)
                if not isinstance(sigs, list) and not isinstance(sigs, tuple):
                    sigs = [sigs]
                # Create a collapsible container for each signature
                for sig in sigs:
                    if isinstance(sig, DistributionSignature):
                        # Plot the signature
                        methods_ct.append(sig.html())
                    else:
                        methods_ct.append(pg.Text(f"No data for {method}"))
            else:
                methods_ct.append(pg.Text(f"Unknown method: {method}"))
        
        # Now let's add the comparison between plots
        # Create a container for the comparison
        comparison_ct = pg.CollapsibleContainer("⚖️ Comparison", layout='vertical')
        # Add the comparison container to the main container
        obj_ct.append(comparison_ct)

        """ 
            Correlation plot 
        """
        # Add one container for the correlation
        corr_ct = pg.CollapsibleContainer("📊 Correlation", layout='vertical')

        # Add a non-collapsible container to make it look good
        if 'plot_correlation' in netsurf.doc.uncertainty.UNCERTAINTY_PLOTS_DOC:
            corr_explanation = netsurf.doc.uncertainty.UNCERTAINTY_PLOTS_DOC['plot_correlation']
            corr_exp_ct = pg.Container("Values explained", layout='vertical', attributes={'style': 'padding: 10px;'})

            # Add a markdown text explaining the information that we get from here:
            corr_explanation = pg.Markdown(corr_explanation, attributes={'style': 'padding: 10px;'})
            
            # Add the explanation to the container
            corr_exp_ct.append(corr_explanation)
            # Add the explanation container to the correlation container
            corr_ct.append(corr_exp_ct)

        # Add correlation plot
        corr_fig, corr_axs = plt.subplots(2,1,figsize=(8, 16))
        _, _ = self.plot_correlations(sample_size=-1, axs = corr_axs, show = False)
        # Convert the plot to an image
        plt.rcParams['text.usetex'] = True
        img = pg.Image(corr_fig, embed=True)
        plt.rcParams['text.usetex'] = False
        # Add the image to the container
        corr_ct.append(img)
        # Add the correlation container to the comparison container
        comparison_ct.append(corr_ct)


        """ 
            Radar plots 
        """
        # Add one container for the radar plots
        radar_ct = pg.CollapsibleContainer("🕸️ Radar", layout='vertical')
        # Add a non-collapsible container to make it look good
        if 'plot_radar_fingerprints' in netsurf.doc.uncertainty.UNCERTAINTY_PLOTS_DOC:
            radar_explanation = netsurf.doc.uncertainty.UNCERTAINTY_PLOTS_DOC['plot_radar_fingerprints']
            radar_exp_ct = pg.Container("Values explained", layout='vertical', attributes={'style': 'padding: 10px;'})

            # Add a markdown text explaining the information that we get from here:
            radar_explanation = pg.Markdown(radar_explanation, attributes={'style': 'padding: 10px;'})
            
            # Add the explanation to the container
            radar_exp_ct.append(radar_explanation)
            # Add the explanation container to the correlation container
            radar_ct.append(radar_exp_ct)
        # Add radar plot
        radar_fig, radar_axs = self.plot_radar_fingerprints(axs = None, 
                                                            show = False)
        # Convert the plot to an image
        plt.rcParams['text.usetex'] = True
        img = pg.Image(radar_fig, embed=True)
        plt.rcParams['text.usetex'] = False
        # Add the image to the container
        radar_ct.append(img)
        # Add the radar container to the comparison container
        comparison_ct.append(radar_ct)



        return obj_ct
    


@dataclass
class ProfileDivergence:
    divergences: Dict[str, RobustnessDivergence]

    @staticmethod
    def from_signatures(pre: RobustnessSignature, post: RobustnessSignature, methods=None) -> "ProfileDivergence":
        """
        Compares two RobustnessSignatures and returns divergence metrics per method.
        """
        if methods is None:
            methods = ["fisher", "aleatoric", "delta", "qpolar", "qpolargrad"]

        divergences = {}
        for method in methods:
            if hasattr(pre, method) and hasattr(post, method):
                pre_sig = getattr(pre, method)
                post_sig = getattr(post, method)
                if isinstance(pre_sig, DistributionSignature) and isinstance(post_sig, DistributionSignature):
                    divergences[method] = RobustnessDivergence.from_signatures(pre_sig, post_sig)

        return ProfileDivergence(divergences=divergences)
    
    def plot_divergence_summary(self, figsize=(10, 5), show=True):
        """
        Visualizes robustness divergence between pre- and post-training distributions.
        Includes:
        - Cosine similarity (bar chart)
        - Gini coefficient shift (line plot on secondary axis)
 
        Args:
            figsize (tuple): Size of the figure.
            show (bool): Whether to display the plot immediately.
        """
 
        methods = list(self.divergences.keys())
        cosine_sims = [self.divergences[m].cosine_similarity for m in methods]
        gini_shifts = [self.divergences[m].gini_shift for m in methods]
 
        fig, ax1 = plt.subplots(figsize=figsize)
 
        color1 = 'tab:blue'
        ax1.set_xlabel('Method')
        ax1.set_ylabel('Cosine Similarity', color=color1)
        bars = ax1.bar(methods, cosine_sims, color=color1, alpha=0.7, label='Cosine Similarity')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.set_ylim(0, 1.05)
 
        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel('Gini Coefficient Shift', color=color2)
        ax2.plot(methods, gini_shifts, color=color2, marker='o', label='Gini Shift')
        ax2.tick_params(axis='y', labelcolor=color2)
 
        fig.tight_layout()
        if show:
            plt.show()
 
        return fig, (ax1, ax2)

    def plot_advanced_divergence_summary(self, figsize=(12, 6), show=True):
        """
        Visualizes additional divergence metrics between pre- and post-training profiles:
        - Cosine Similarity
        - Gini Shift
        - KL Divergence (symmetrized)
        """
        methods = list(self.divergences.keys())
        cosine_sims = []
        gini_shifts = []
        kl_divs = []

        for m in methods:
            div = self.divergences[m]
            cosine_sims.append(div.cosine_similarity)
            gini_shifts.append(div.gini_shift)

            pre = div.distribution_stats.get("pre", None)
            post = div.distribution_stats.get("post", None)
            if pre is not None and post is not None:
                pre_pdf = np.abs(pre) / (np.sum(np.abs(pre)) + 1e-8)
                post_pdf = np.abs(post) / (np.sum(np.abs(post)) + 1e-8)
                kl_1 = kl_divergence(pre_pdf + 1e-8, post_pdf + 1e-8)
                kl_2 = kl_divergence(post_pdf + 1e-8, pre_pdf + 1e-8)
                kl_divs.append(0.5 * (kl_1 + kl_2))  # symmetric KL
            else:
                kl_divs.append(0.0)

        fig, ax1 = plt.subplots(figsize=figsize)

        ax1.set_xlabel("Method")
        ax1.set_ylabel("Cosine Similarity / Gini Shift", color='tab:blue')
        ax1.bar(methods, cosine_sims, label='Cosine Similarity', color='tab:blue', alpha=0.6)
        ax1.plot(methods, gini_shifts, label='Gini Shift', color='tab:orange', marker='o')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.set_ylim(0, 1.1)

        ax2 = ax1.twinx()
        ax2.set_ylabel("Symmetric KL Divergence", color='tab:red')
        ax2.plot(methods, kl_divs, label='KL Divergence', color='tab:red', marker='x')
        ax2.tick_params(axis='y', labelcolor='tab:red')

        fig.tight_layout()
        if show:
            plt.show()
        return fig, (ax1, ax2)


class UncertaintyProfiler:
    """
    Computes various sources of uncertainty and robustness-related statistics
    for a given model and dataset. Returns a populated RobustnessSignature.
    """
    
    @staticmethod
    def compute_fisher_distribution(model, dataset, loss_fn, batch_size = 32, verbose = True, **kwargs) -> DistributionSignature:
        """
        Approximates the Fisher Information diagonal via squared gradients over the validation set.
        Returns a DistributionSignature summarizing the distribution of Fisher values.
        """
        fisher_diags = None
        # Get num samples 
        X, Y = dataset 
        n_samples = X.shape[0]
        n_batches = int(n_samples // batch_size)

        # Init a progress bar
        pbar = netsurf.utils.io.ProgressBar(total=n_batches, prefix='Computing Fisher diagonal')

        # Loop thru batches
        for i in range(n_batches):
            # Get batch
            x_batch = X[i * batch_size:(i + 1) * batch_size]
            y_batch = Y[i * batch_size:(i + 1) * batch_size]
            # Compute fisher diagonal
            if verbose: pbar.update(i+1)
            # Compute gradients
            with tf.GradientTape() as tape:
                preds = model(x_batch, training=False)
                loss = loss_fn(y_batch, preds)
            grads = tape.gradient(loss, model.trainable_variables)
            if fisher_diags is None:
                fisher_diags = [tf.square(g) if g is not None else None for g in grads]
            else:
                fisher_diags = [fd + tf.square(g) if fd is not None and g is not None else fd
                                for fd, g in zip(fisher_diags, grads)]
        
        if not fisher_diags:
            return None

        # Average the accumulated gradients over batches
        fisher_diags = [fd / n_batches if fd is not None else None for fd in fisher_diags]

        # Flatten and concatenate all fisher diagonals
        all_fisher = tf.concat([tf.reshape(fd, [-1]) for fd in fisher_diags if fd is not None], axis=0)

        # Compute the trace
        fisher_trace = tf.reduce_sum(all_fisher).numpy()

        dis = DistributionSignature.from_array(all_fisher.numpy(), 
                                                name="fisher", 
                                                extra_fields={'trace': fisher_trace},
                                                color = DEFAULT_DISTRIBUTION_COLORS['fisher'],
                                                plot_functions = DEFAULT_PLOT_FUNCTIONS['fisher'],
                                                is_data_pdf = True)

        return dis

    @staticmethod
    def compute_aleatoric_distribution(model, dataset, loss_fn=None, batch_size=32, **kwargs) -> DistributionSignature:
        """
        Computes aleatoric uncertainty by measuring the gradient norm of the loss with respect to inputs.
        Returns a DistributionSignature of ∇_X L norms.
        """
        X, Y = dataset
        n_samples = X.shape[0]
        n_batches = int(n_samples // batch_size)
 
        grad_norms = []
 
        pbar = netsurf.utils.io.ProgressBar(total=n_batches, prefix='Computing Aleatoric Uncertainty')
        for i in range(n_batches):
            x_batch = X[i * batch_size:(i + 1) * batch_size]
            y_batch = Y[i * batch_size:(i + 1) * batch_size]
            pbar.update(i + 1)
            # Convert to tf.Tensor
            x_batch = tf.convert_to_tensor(x_batch, dtype=tf.float32)
            y_batch = tf.convert_to_tensor(y_batch, dtype=tf.float32)
            with tf.GradientTape() as tape:
                tape.watch(x_batch)
                preds = model(x_batch, training=False)
                loss = loss_fn(y_batch, preds) if loss_fn else tf.reduce_mean(preds)
            grads = tape.gradient(loss, x_batch)
            norm = tf.norm(tf.reshape(grads, (grads.shape[0], -1)), axis=-1)
            grad_norms.append(norm)
 
        grad_norms = tf.concat(grad_norms, axis=0).numpy()
        return DistributionSignature.from_array(grad_norms, 
                                                name="aleatoric", 
                                                plot_functions = DEFAULT_PLOT_FUNCTIONS['aleatoric'],
                                                color = DEFAULT_DISTRIBUTION_COLORS['aleatoric'])

    @staticmethod
    def compute_msb_impact_ratio(model, dataset, batch_size = 128, uncorrupted_impacts = None, **kwargs) -> float:
        """
        Computes the MSB (Most Significant Bit) impact ratio, which estimates
        how much of the total bit-level perturbation impact is concentrated in
        the MSBs of quantized weights.
        This function assumes that each custom quantized layer exposes a
        `bit_importance()` method returning a tensor of shape [..., n_bits].
        """
        print("Computing MSB impact ratio...")
        # We actually care about the ratio msb_impacts / total_impacts
        msb_impacts = []
        total_impacts = []
        msb_impacts_abs = []
        total_impacts_abs = []
        msb_over_total_ratios_signed = []
        msb_over_total_ratios_abs = []

        # If uncorrupted impacts are None, we need to compute them
        if uncorrupted_impacts is None:
            uncorrupted_impacts, corrupted_impacts = UncertaintyProfiler.compute_impacts(model, dataset, batch_size = batch_size, **kwargs)

        # Now simply loop thru them and compute the ratios
        for kw, kv in uncorrupted_impacts.items():
            # Get the MSB impact
            msb_impact = kv[..., 0]
            msb_impact_abs = tf.abs(msb_impact)
            total_impact = tf.reduce_sum(kv, axis=-1)
            total_impact_abs = tf.abs(tf.reduce_sum(kv, axis=-1))

            # Compute the ratio per weight
            ratio = msb_impact / (total_impact + 1e-8)
            ratio_abs = msb_impact_abs / (total_impact_abs + 1e-8)

            # Reshape
            msb_impact = tf.reshape(msb_impact, [-1])
            msb_impact_abs = tf.reshape(msb_impact_abs, [-1])
            total_impact = tf.reshape(total_impact, [-1])
            total_impact_abs = tf.reshape(total_impact_abs, [-1])
            ratio = tf.reshape(ratio, [-1])
            ratio_abs = tf.reshape(ratio_abs, [-1])


            # Add to the list
            msb_impacts.append(msb_impact)
            msb_impacts_abs.append(msb_impact_abs)
            total_impacts.append(total_impact)
            total_impacts_abs.append(total_impact_abs)

            # compute the ratio per weight
            msb_over_total_ratios_signed.append(ratio)
            msb_over_total_ratios_abs.append(ratio_abs)


        # Get 6 distribution signatures, one for signed and one for absolute
        msb_impacts = tf.concat(msb_impacts, axis=0)
        msb_impacts_abs = tf.concat(msb_impacts_abs, axis=0)
        total_impacts = tf.concat(total_impacts, axis=0)
        total_impacts_abs = tf.concat(total_impacts_abs, axis=0)
        msb_over_total_ratios_signed = tf.concat(msb_over_total_ratios_signed, axis=0)
        msb_over_total_ratios_abs = tf.concat(msb_over_total_ratios_abs, axis=0)

        # Compute the distribution signatures
        msb_impact_signature = DistributionSignature.from_array(msb_impacts.numpy(), name="msb_impact", plot_functions = DEFAULT_PLOT_FUNCTIONS['msb'], color = DEFAULT_DISTRIBUTION_COLORS['msb'])
        total_impact_signature = DistributionSignature.from_array(total_impacts.numpy(), name="total_impact", plot_functions = DEFAULT_PLOT_FUNCTIONS['msb'], color = DEFAULT_DISTRIBUTION_COLORS['msb'])
        msb_impact_abs_signature = DistributionSignature.from_array(msb_impacts_abs.numpy(), name="msb_impact_abs", plot_functions = DEFAULT_PLOT_FUNCTIONS['msb'], color = DEFAULT_DISTRIBUTION_COLORS['msb'])
        total_impact_abs_signature = DistributionSignature.from_array(total_impacts_abs.numpy(), name="total_impact_abs", plot_functions = DEFAULT_PLOT_FUNCTIONS['msb'], color = DEFAULT_DISTRIBUTION_COLORS['msb'])
        msb_over_total_ratios_signed = DistributionSignature.from_array(msb_over_total_ratios_signed.numpy(), name="msb_over_total_signed", plot_functions = DEFAULT_PLOT_FUNCTIONS['msb'], color = DEFAULT_DISTRIBUTION_COLORS['msb'])
        msb_over_total_ratios_abs = DistributionSignature.from_array(msb_over_total_ratios_abs.numpy(), name="msb_over_total_abs", plot_functions = DEFAULT_PLOT_FUNCTIONS['msb'], color = DEFAULT_DISTRIBUTION_COLORS['msb'])

        # Return as a dict
        d = {'msb_impact_ratio': msb_over_total_ratios_signed,
             'msb_impact_ratio_abs': msb_over_total_ratios_abs,
             'msb_impact': msb_impact_signature,
             'total_impact': total_impact_signature,
             'msb_impact_abs': msb_impact_abs_signature,
             'total_impact_abs': total_impact_abs_signature}

        return d

    @staticmethod
    def compute_delta_distribution(model, **kwargs) -> DistributionSignature:
        """
        Computes the distribution of delta values across all quantized layers in the model.
        This reflects how much a bit-flip in each position changes the associated weight.
        """
        print("Computing delta distribution...")
        if not hasattr(model, "deltas"):
            return None
        # Get the deltas
        deltas = model.deltas

        # Flatten and concatenate all deltas
        deltas = np.concatenate([d.flatten() for d in deltas if d is not None], axis=0)

        # Deltas can only have very specific values do to the quantization
        bins = model.quantizer.qbins

        return DistributionSignature.from_array(deltas, 
                                                name="delta", 
                                                plot_functions = DEFAULT_PLOT_FUNCTIONS['delta'],
                                                color=DEFAULT_DISTRIBUTION_COLORS["delta"], 
                                                bins = bins, 
                                                quantized=True)

    @staticmethod
    def compute_qpolar_distribution(model, dataset, batch_size = 128, corrupted_impacts = None, **kwargs) -> DistributionSignature:
        """
        Computes the qpolar impact ratio
        """
        print("Computing qpolar distribution...")
        # We actually care about the ratio msb_impacts / total_impacts
        qpolar_impacts = []

        # If uncorrupted impacts are None, we need to compute them
        if corrupted_impacts is None:
            uncorrupted_impacts, corrupted_impacts = UncertaintyProfiler.compute_impacts(model, dataset, batch_size = batch_size, **kwargs)

        # Qpolar is just the absolute value of the impact
        for kw, kv in corrupted_impacts.items():
            # Get the MSB impact
            total_impact_abs = tf.abs(kv)
            # Reshape
            total_impact_abs = tf.reshape(total_impact_abs, [-1])
            # Append
            qpolar_impacts.append(total_impact_abs)

        # Concat
        qpolar_impacts = tf.concat(qpolar_impacts, axis=0)
        
        # Compute the distribution signatures
        return DistributionSignature.from_array(qpolar_impacts.numpy(), 
                                                name="qpolar", 
                                                plot_functions = DEFAULT_PLOT_FUNCTIONS['qpolar'],
                                                color = DEFAULT_DISTRIBUTION_COLORS['qpolar'])
        

    @staticmethod
    def compute_qpolargrad_distribution(model, dataset, loss_fn, batch_size = 128, corrupted_impacts = None, **kwargs) -> DistributionSignature:
        """
        Computes the qpolar impact ratio
        """
        print('Computing qpolargrad distribution')

         # If uncorrupted impacts are None, we need to compute them
        if corrupted_impacts is None:
            uncorrupted_impacts, corrupted_impacts = UncertaintyProfiler.compute_impacts(model, dataset, batch_size = batch_size, **kwargs)

        # Get data for gradient computation
        X, Y = dataset
        
        # Now we need to get the gradients
        # We will use the activation model to get the output for every layer 
        with tf.GradientTape(persistent = True) as tape:
            # Forward pass
            predictions = model(X, training=True)
            
            # Calculate loss
            loss = loss_fn(Y, predictions)
            
            # Add regularization losses if any
            if model.losses:
                loss += tf.math.add_n(model.losses)
        
        # Get the gradients
        grads = tape.gradient(loss, model.trainable_variables)

        # Convert into dict
        grads = {k.name: g for k, g in zip(model.trainable_variables, grads)}

        # Now we need to multiply the gradients with the qpolar impacts
        qpolargrad_impacts = []
        for varname, qpolar_impact in corrupted_impacts.items():
            # Get the corresponding gradient
            grad = grads[varname]
            if grad is not None:
                # Add axis to grad to match the qpolar impact
                grad = grad[...,tf.newaxis]

                # Compute the impact
                qpolargrad_impacts.append(tf.reshape(tf.abs(tf.multiply(qpolar_impact, grad)), [-1]))

        # Get 6 distribution signatures, one for signed and one for absolute
        qpolargrad_impacts = tf.concat(qpolargrad_impacts, axis=0)

        # Compute the distribution signatures
        return DistributionSignature.from_array(qpolargrad_impacts.numpy(), 
                                                name="qpolargrad", 
                                                plot_functions = DEFAULT_PLOT_FUNCTIONS['qpolargrad'],
                                                color = DEFAULT_DISTRIBUTION_COLORS['qpolargrad'])

    @staticmethod
    def compute_empirical_metrics(model, dataset, method, **kwargs) -> Tuple[float, float]:
        # TODO: run BER test and return (accuracy, ranking effectiveness)
        return None, None
    
    @staticmethod
    def compute_impacts(model, dataset, batch_size = 128, **kwargs) -> Dict[str, DistributionSignature]:
        """
        Computes the impacts for all layers in the model.
        Returns a dictionary of impact distributions.
        """
        print("Computing impacts...")
        # We actually care about the ratio msb_impacts / total_impacts
        corrupted_impacts = {}
        uncorrupted_impacts = {}

        # Get dataset and batch size
        X, Y = dataset
        
        # Compute impacts 
        # Let's get the activation for each layer BUT with full corruption (N=1)
        uncorrupted_output, uncorrupted_activations = model.attack(X, N = 0, return_activations = True)
        corrupted_output, corrupted_activations = model.attack(X, N = 1, return_activations = True)

        for ily, ly in enumerate(model.layers):
            if not hasattr(ly, 'attack'):
                continue 

            # Get the input_tensor name 
            input_tensor = ly.input.name.rsplit('/',1)[0]

            # If we can find it in the activations, we can compute the impact
            if input_tensor not in uncorrupted_activations or input_tensor not in corrupted_activations:
                continue
            # Get act
            uncorrupted_act = uncorrupted_activations[input_tensor]
            corrupted_act = corrupted_activations[input_tensor]

            if hasattr(ly, 'compute_impact'):
                # Just compute the impact by directly calling the layer's method 
                uncorrupted_impacts = {**uncorrupted_impacts, **ly.compute_impact(uncorrupted_act, batch_size = batch_size)}
                corrupted_impacts = {**corrupted_impacts, **ly.compute_impact(corrupted_act, batch_size = batch_size)}

        return uncorrupted_impacts, corrupted_impacts

    @staticmethod
    def profile(model, dataset, loss_fn, methods=None, **kwargs) -> RobustnessSignature:
        """
        Generates a complete robustness signature for the given model and evaluation data.
        """
        if methods is None:
            methods = ["fisher", "aleatoric", "msb", "delta", "qpolar", "qpolargrad"]
        # Initialize 
        dists = {'empirical_ber_accuracy': None, 'ranking_effectiveness': None}
        
        # if we are gonna get msb/qpolar/qpolargrad, we will need the impact computation for all of them
        # so instead of computing it over and over again, we will just compute it once here.
        if 'msb' in methods or 'qpolar' in methods or 'qpolargrad' in methods:
            uncorrupted_impacts, corrupted_impacts = UncertaintyProfiler.compute_impacts(model, dataset, **kwargs)
            # add to kwargs
            kwargs['uncorrupted_impacts'] = uncorrupted_impacts
            kwargs['corrupted_impacts'] = corrupted_impacts


        # Loop thru methods
        methods_caller = {'fisher': lambda m,d,l,**kws: UncertaintyProfiler.compute_fisher_distribution(m, d, l, **kws),
                          'aleatoric': lambda m,d,l,**kws: UncertaintyProfiler.compute_aleatoric_distribution(m, d, l, **kws),
                          'msb': lambda m,d,*args,**kws: UncertaintyProfiler.compute_msb_impact_ratio(m, d, **kws),
                          'delta': lambda m,*args,**kws: UncertaintyProfiler.compute_delta_distribution(m, **kws),
                          'qpolar': lambda m,d,*args,**kws: UncertaintyProfiler.compute_qpolar_distribution(m, d, **kws),
                          'qpolargrad': lambda m,d,l,**kws: UncertaintyProfiler.compute_qpolargrad_distribution(m, d, l, **kws)
                          }
        
        translate = {'fisher': 'fisher_distribution',
                      'aleatoric': 'aleatoric_distribution',
                      'msb': 'msb_impact_ratio',
                      'delta': 'delta_distribution',
                      'qpolar': 'qpolar_distribution',
                      'qpolargrad': 'qpolargrad_distribution',
                      }
        for method in methods:
            # lowercase
            method = method.lower()
            t = translate.get(method, method)
            # Check if method name is in kwargs (maybe the user is literally passing the result object)
            if method in kwargs:
                dists[t] = kwargs.pop(method)
            elif t in kwargs:
                dists[t] = kwargs.pop(t)
            elif method in methods_caller:
                # Remember that In reality, qpolar is just the impact (total_impacts_abs) (msb)
                # so if we already have it, we can just return it
                res = methods_caller[method](model, dataset, loss_fn, **kwargs)
                if isinstance(res, dict):
                    # Unpack the msb impact ratio
                    dists = {**dists, **res}
                else:
                    dists[t] = res
            else:
                raise ValueError(f"Unknown method: {method}")
        
        return RobustnessSignature(**dists)
    

    
    