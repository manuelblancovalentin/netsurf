""" This document just holds some markdown text for keeping track of the 
    documentation for some metrics and uncertainty quantification methods.
    We'll use these in our reports with pergamos
    """

UNCERTAINTY_METRICS_DOC_LORENZ_PLOT = """
## 🔎 What the Lorenz Curve Shows
 * X-axis: Cumulative proportion of parameters (e.g. 0.8 means 80% of least influential parameters).
 * Y-axis: Cumulative proportion of total “importance mass” (e.g. Fisher, qpolar, etc.).

#### 🔹 Interpretation:
- The **diagonal line** represents perfect equality — each element contributes equally.
- The **further below the diagonal** the curve bends, the **more unequal** the distribution is.
- If 20% of elements contribute only 1% of total mass, you’ll see that as a sharp curve.

#### 📊 Use Case:
- Quantifies **how many parameters matter**.
- Visually identifies whether **sensitivity is sparse or dense**.
- Useful for deciding if ranking or pruning strategies will be effective.

#### 🧮 Bonus: Gini Coefficient
- Area between the diagonal and the Lorenz curve → **Gini index** (0 = equality, 1 = total inequality).
    
#### 🧠 Gini Coefficient as a Ranking Proxy
 * The Gini coefficient quantifies inequality — in our context, the inequality of importance/sensitivity across parameters.
 * A high Gini value means that a few parameters dominate the total “importance mass” (e.g., Fisher, QPolarGrad).
 * If a distribution is highly concentrated (Gini ≈ 1), then ranking by that metric should be very effective, because:
 * You only need to protect a few parameters to prevent large damage.
 * The signal is clear — it's easy to identify what to rank.
 * In contrast, a low Gini value means importance is more spread out, and ranking becomes harder:
 * No obvious top parameters.
 * Even large groups may carry moderate amounts of mass.

> This explains why QPolarGrad often outperforms QPolar:
>  * QPolarGrad has a much higher Gini (near 1), so it's more selective and concentrated.
>  * QPolar is more distributed, suggesting that while it's informative, its effectiveness depends on broader statistical behaviors — e.g., it's capturing subtler structural aspects.

#### 🤔 What does it mean if the gini is ~ 1.0
Gini ≈ 0.9997–0.9998
* Nearly all importance is concentrated in very few parameters.
* Curve is flat until it jumps up at the very end — like 0% → 100% mass in just a few top-ranked parameters.
* This means that bit-flipping most weights will have near-zero impact, while flipping a few is catastrophic.
* Very pruning- or ranking-friendly — you can remove 99% of weights without touching the critical ones.
* From a robustness engineering perspective, this is a blessing and a curse:
* ✅ Great for defenses based on ranking.
* ❌ Dangerous if a random flip hits a critical parameter.

"""

UNCERTAINTY_METRICS_DOC_PERCE_PLOT = """
## 🔀 Perce Curve

The **Perce Curve** is a Lorenz-like plot, but **re-centered** around the 50% mass point.

- **X-axis**: Parameter rank, centered around the point where 50% of the mass has accumulated.
- **Y-axis**: Cumulative contribution (normalized sum of sorted values).

#### 🔹 Interpretation:
- A steep initial rise shows that **a small number of parameters dominate early**.
- The **X=0** line indicates where **half the total mass** is concentrated.

#### 🧠 Why it's useful:
- Easier to see how many parameters are above/below the 50% mark.
- Highlights **symmetry or skew** in importance distribution.
- Useful when comparing models or metrics with different sharpness.

"""

UNCERTAINTY_METRICS_DOC_CUMULATIVE_MASS_PLOT = """
## 📈 Cumulative Mass Plot

This plot shows **how quickly the total "mass" (e.g., importance or energy)** accumulates across sorted parameters.

- **X-axis**: Rank of parameters (sorted by importance).
- **Y-axis**: Cumulative mass (sum of values, normalized).

#### 🔹 Interpretation:
- If the curve reaches 80% mass quickly, that means **few parameters dominate**.
- A slower climb implies a **more evenly distributed** influence.

#### 📊 Use Case:
- Lets you **estimate how many parameters you need to cover** most of the distribution.
- Great for **comparing distributions** to see which is more sparse or peaky.
"""





UNCERTAINTY_METRICS_DOC_CORRELATION_PLOT = """### 🔍 Interpreting the Correlation Plots

These heatmaps visualize the **Pearson correlation coefficients** between either raw distribution values or summary statistics of uncertainty signatures.

---

#### 📊 Plot 1: Correlation of Raw Distributions

Each cell \((i, j)\) in this matrix quantifies **how linearly correlated** the actual *values* of two uncertainty distributions are:

- **Value near +1** → Strong positive linear relationship. When one metric has a high value, so does the other.
- **Value near 0** → No linear relationship. The distributions are **independent** in shape and magnitude.
- **Value near -1** → Strong inverse linear relationship (rare in these metrics).

**Why it matters**:
- If values are highly correlated, the distributions might be **redundant** — they carry overlapping information.
- Low or near-zero correlations mean the metrics are **orthogonal**, i.e., each captures a unique structural or probabilistic property.

In most robust uncertainty models, we *want* these metrics to be uncorrelated — that means each one contributes something novel.

---

#### 📈 Plot 2: Correlation of Summary Statistics

This plot shows how similar the **summary descriptors** of distributions are — such as:

- Entropy (spread/unpredictability),
- Variance (spread),
- Skewness (asymmetry),
- Kurtosis (peakedness),
- Mean, std, min/max, and energy norms.

**Interpretation**:

- **High correlation (≥ 0.9)** between, e.g., `entropy` and `variance`, implies that if one goes up, the other tends to as well — likely due to shared mathematical properties.
- **Lower correlation** values (especially with `min` or `max`) indicate that some statistics may respond differently across distributions — useful for identifying *diverse shape features*.

**Why this is useful**:
- Helps detect which statistics are **redundant** and could be **dropped** to simplify profiling.
- A high-correlation block implies that you might compress dimensionality via PCA or clustering.
- A low-correlation statistic (like `min`) may be **uniquely informative** and worth preserving.

---

#### ✅ TL;DR Summary

| Plot                        | Tells You About...                         | Desired Outcome       |
|-----------------------------|---------------------------------------------|------------------------|
| Raw Distributions           | Structural diversity between uncertainty types | Low correlation (orthogonality) |
| Summary Statistics          | Redundancy or synergy between descriptors   | Use to select minimal informative set |
"""



UNCERTAINTY_METRICS_DOC_RADAR_PLOT = """
## 🕸️ Radar Plot (Statistical Fingerprint)

The **Radar Plot** provides a compact, visual summary of various **statistical descriptors** extracted from a distribution (e.g., Fisher, QPolar, etc.). It creates a “fingerprint” of the shape and spread of the data.

---

### 📐 Axes (Dimensions):
Each axis corresponds to a different summary statistic:
- \[$\mathcal{H}$\] (Entropy) → randomness or unpredictability in the distribution
- \[$\sigma^2$\] (Variance) → overall spread
- \[$\gamma_1$\] (Skewness) → asymmetry
- \[$\gamma_2$\] (Kurtosis) → tail-heaviness
- \[$\mu$\] (Mean) → average value
- \[$\sigma$\] (Standard Deviation) → dispersion
- Min / Max → extreme values
- \[$\|x\|_1$\] (L1 Energy) → total absolute mass
- \[$\|x\|_2^2$\] (L2 Energy) → quadratic energy
- \[$Gini$\] → inequality of distribution
- \[$\mathcal{L}$\] (Lorenz) → cumulative mass distribution
- \[$\mathcal{P}$\] (Perce) → cumulative mass distribution centered around 50% mass

Each value is **normalized** across all distributions being compared (min-max or z-score), so they are plotted on a common scale (typically [0,1]).

---

### 🔹 Interpretation:

- **Spiky patterns**: Sharp spikes suggest a distribution dominates on a particular statistic.
- **Rounder shapes**: More balanced or uniform statistic values.
- **Overlap**: If plotted together, similar shapes = similar statistical profile.
- **Outliers**: A method with extreme skewness or entropy will clearly stand out.

---

### 🧠 Why it's useful:

- Enables **quick visual comparison** across different distributions.
- Highlights **which properties dominate** for each method (e.g., QPolar vs Fisher).
- Detects **unique statistical shapes** that may not be evident from histograms alone.
- Useful for **debugging**, **feature selection**, or method clustering.

---

> TL;DR: Radar plots let you “see the personality” of a distribution at a glance.

"""








# Create a dictionary
# to hold the markdown text for each metric
UNCERTAINTY_PLOTS_DOC = {'plot_lorenz_curve': UNCERTAINTY_METRICS_DOC_LORENZ_PLOT,
                         'plot_perce_curve': UNCERTAINTY_METRICS_DOC_PERCE_PLOT,
                         'plot_cumulative_mass': UNCERTAINTY_METRICS_DOC_CUMULATIVE_MASS_PLOT,
                         'plot_correlation': UNCERTAINTY_METRICS_DOC_CORRELATION_PLOT,
                         'plot_radar_fingerprints': UNCERTAINTY_METRICS_DOC_RADAR_PLOT}