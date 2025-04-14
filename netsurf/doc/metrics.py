ANALYSIS_METRIC_DOC_MOMENTS = r"""
## 📈 Statistical Moment Grids (Mean, Std, Skewness, Kurtosis, Count)

For each metric (e.g., accuracy, loss), we can compute local **distributional statistics** over the grid by aggregating repeated Monte Carlo runs at each `(protection, BER)` point.

This results in **moment maps** — one per statistical measure — that provide richer insights than just the average.

---

### 🧮 Computed Moments

| Moment         | Description |
|----------------|-------------|
| **mean**       | The average of the metric (e.g., categorical accuracy) across all repetitions at this grid cell. Represents *expected* model performance under bit-flip randomness. |
| **std**        | Standard deviation — how much the model's performance *fluctuates* across repetitions. High values indicate *sensitivity to randomness* in bit flips. |
| **skewness**   | Measures *asymmetry* in the distribution of outcomes. Positive skew = occasional very good outcomes, negative = occasional catastrophic drops. Helps detect *hidden risks*. |
| **kurtosis**   | Measures *tail heaviness*. High kurtosis implies a greater likelihood of *rare extreme cases* (instability, failure, etc.). |
| **count**      | Number of valid samples aggregated in each cell. Useful for confidence weighting or identifying under-sampled regions. |

---

### 🧪 Why Moments Matter

These statistical maps allow you to answer nuanced questions such as:
- *Where is the model fragile even if the average performance is high?*
- *Which regions are highly predictable, and which are chaotic?*
- *Are there regions where one bad flip causes disproportionately bad behavior (skewed)?*
- *Where should I invest more sampling effort (low count, high variance)?*

---

### 🔁 Advanced Use

These moments can also be treated as scalar fields and passed to:

- `compute_surface_fields` → to compute **gradient of std**, **laplacian of skewness**, etc.
- Derive *uncertainty topology*, *sensitivity ridges*, or *robustness hotspots*.

In essence, you are not just mapping performance — you are mapping the **shape of robustness**.

---

### 🚨 Best Practice

While the **mean** field is the baseline, combining it with **std**, **skew**, and **kurtosis** reveals:
- **Where to trust the mean**
- **Where flip-based volatility is high**
- **Where rare but critical failures may happen**

These are essential for **safety-critical**, **defense-oriented**, or **reliability-first** applications.
"""



ANALYSIS_METRIC_DOC_FIELDS = r"""
## 🔍 Field Analysis Overview

When analyzing the performance of different ranking methods under bit-flip attacks in a 2D space (Protection × BER), we can go beyond average accuracy and compute **differential properties of the surface**. These reveal how robustness **evolves**, **spikes**, or **collapses** across the grid.

---

### 🧮 `Surface Fields`

This function computes a set of **local descriptors** (mathematical fields) from the accuracy grid.

| Field        | Description |
|--------------|-------------|
| **mean**     | The interpolated accuracy surface at each (protection, BER) point. |
| **gradient** | Magnitude of local slope — shows how sharply accuracy changes. High values = sensitive regions. |
| **laplacian**| Sum of curvature — detects performance "sources" and "sinks". Positive = bowl-like, Negative = hill-like. |
| **eigen1, eigen2** | Principal curvatures from the Hessian. Describe how accuracy curves along dominant directions. |
| **det**      | Hessian determinant. Negative = saddle point, Positive = extremum. Useful for detecting volatile regions. |
| **condition**| Ratio of eigenvalues. High = accuracy changes sharply in one direction but is flat in the other. Measures **anisotropy** of sensitivity. |

These fields help map how **fragile** or **stable** the model is in different regions — beyond average performance.

---

### 📊 `Barycentric Strength`

We stack the accuracy grids of each ranker and normalize them so that each ranker contributes proportionally:

$begin:math:display$
Z_{\\text{bary}}[i] = \\frac{Z[i]}{\\sum_j Z[j]}
$end:math:display$

- Values close to 1 → ranker *i* dominates.
- Values ≈ 1/N → evenly matched (N = number of rankers).
- These maps form a **probability simplex** across rankers and locations.

---

### ✅ `Trust overlay`

Not all dominance is meaningful — sometimes a ranker "wins" by just a tiny margin. To highlight *trustworthy* regions:

- We threshold the barycentric strengths at 1/N (default: 1/3 for 3 rankers).
- This produces a **mask** of confident dominance.
- The **trust overlay** multiplies the mask by the strength → so we see both region and intensity.
- The **coverage** quantifies how much of the space each ranker truly dominates.

---

### 🌈 `RGB Dominance`

This produces a **continuous RGB visualization** of ranker dominance:

- Center the Z values (subtract mean per point) → focus on relative, not absolute, performance.
- Normalize → encode direction of dominance, not its magnitude.
- Use ranker-assigned RGB colors to blend each pixel based on who dominates there.

This produces a **smooth visual map** that intuitively communicates complex interactions between ranking methods across space.
"""



# Create a dictionary
# to hold the markdown text for each metric
ANALYSIS_METRICS_DOC = {'moments': ANALYSIS_METRIC_DOC_MOMENTS, 
                        'fields': ANALYSIS_METRIC_DOC_FIELDS}