# Run R: a fault-rate-aware protection assignment (ResNet-8 first)

Status, 12 November 2026: plan written before any Run R outcome exists. Nothing is launched.
- Stages 0 and 1 are frozen in full here.
- Stages 2 and 3 have their arms, statistics and decision rules frozen here. Their tuning constants and resource
  estimates are fixed in a dated addendum after Stage 0, from held-out training images only, before their launch.
- Stage 4 is an outline and needs its own plan.
- Every stage needs the researcher's go-ahead in the conversation where it is launched.

Code name of the method: `rate_aware_mask`. It extends `mean_variance_mask` (Run P). It is not `legacy_qpolargrad`,
and the name used in the paper is still open.

Notation note (12 November 2026, before any Run R outcome): formulas were rewritten in LaTeX. No design choice,
number or rule was changed by that rewrite.

## 1. Why

What exists, all exploratory and post hoc (`BER_AWARE_RULE_EXPLORATORY_2026-11-12.md`):
- Run Q showed that the preferred gradient ranking depends on the bit-error probability $p$: the positive gradient
  leads at $10^{-3}$ and falls far behind at $10^{-2}$.
- A two-term formula for the expected loss predicts where that switch happens, from calibration-side images only.
  Its sign matched the Run Q measurement in 74 of 82 usable cells and in all 56 resolved ones.
- Run P's single-rate mask, which minimizes that formula, was never more than 0.42 points below the better of the
  two static rankings in 82 usable cells.

None of this was planned before the data existed, and one definition (the range on which the predicted switch is
located) was changed after output was seen. Run R tests the same statements on fresh fault fields with every
prediction and rule fixed first.

Questions, in order:
1. Does the formula predict the switch? (Stage 1)
2. Is a mask that minimizes the formula never far behind the best ranking at any rate? (Stage 2)
3. Do the residual-error curves of real protection schemes hold, and does choosing the scheme per bit help? (Stage 3)
4. Does any of this carry to other models? (Stage 4)

## 2. Method

**Setup.** Bit $b$ belongs to weight $w$ and moves it by $\Delta_b$ when flipped (known from the stored code). Each
bit has a scheme $s(b)$ with a storage cost $c_s$ and a residual flip probability $\rho_s(p)$ after correction.
Every stored bit, including copies and check bits, flips independently with probability $p$. Voters and decoders
are ideal.

| Scheme | $\rho_s(p)$ | Extra storage |
|---|---|---|
| None | $p$ | 0 |
| Ideal suppression (the paper's masks) | $0$ | counted as 1 bit per bit, as in the paper |
| 3 copies | $3p^2 - 2p^3$ | 2 bits per bit |
| 5 copies | $10p^3 - 15p^4 + 6p^5$ | 4 bits per bit |
| SEC-DED over $k$ data bits, $n$ total | $p\,[1 - (1-p)^{n-1}]$, double errors left as they are, miscorrection ignored | $(n-k)$ per $k$ bits, whole words |
| ABFT, one repair per output channel | measured in Stage 3; no closed form is assumed | two checksums per channel |

**Objective.** For a storage budget $S$ and rate $p$, choose the assignment $s$ that minimizes

$$
J(s;p) = \mathbb{E}_{x}\,\mathbb{E}_{\xi}\big[L_x(\theta + X)\big],
\qquad X_w = \sum_{b \in w} \xi_b\,\Delta_b,
\qquad \xi_b \sim \mathrm{Bernoulli}(r_b),
\qquad r_b = \rho_{s(b)}(p),
$$

subject to

$$
\sum_b c_{s(b)} \le S .
$$

**Analytical part.**

$$
\mu_w = \sum_{b \in w} r_b\,\Delta_b,
\qquad
V_w = \sum_{b \in w} r_b\,(1-r_b)\,\Delta_b^2,
$$

$$
\hat{J}(s;p) = \big[L(\theta+\mu) - L(\theta)\big] + \tfrac12 \sum_w F_w\,V_w,
\qquad
F_w = \mathbb{E}_x\big[g_{x,w}^2\big],
$$

where $g_{x,w}$ is the gradient of the loss of input $x$ with respect to weight $w$.

For codes, $V$ gains the within-word term

$$
\sum_{\substack{b \ne c \\ \text{same word}}} \mathrm{Cov}(\xi_b,\xi_c)\;\mathbb{E}_x\big[e_b\,e_c\big],
\qquad e_b = g_{x,w(b)}\,\Delta_b .
$$

**MC correction.** With $K$ fault fields $X_1,\dots,X_K$,

$$
\hat{C} = \frac1K \sum_{k=1}^{K} \big[L(\theta + X_k) - L(\theta+\mu)\big] - \tfrac12 \sum_w F_w\,V_w .
$$

One uniform number $u_{b,k}$ per stored bit and field is drawn once, and $\xi_{b,k} = \mathbb{1}[u_{b,k} < r_b]$.
The same numbers serve every candidate assignment and every rate.

**Value of changing bit $b$ from scheme $s$ to $s'$, per storage bit.**

$$
v_b = \frac{(r_s - r_{s'})\,\Big[\hat{g}_w\,\Delta_b + \tfrac12\,(1 - r_s - r_{s'})\,F_w\,\Delta_b^2\Big]}{c_{s'} - c_s}
$$

$\hat{g}$ is the gradient at $\theta+\mu$, or the mean signed gradient over the $K$ faulted models,
$\frac1K\sum_k \nabla L(\theta + X_k)$, when the MC correction is on.

**Calibration noise.** Each mean gradient $\bar{g}_w$ is shrunk by its standard error $\mathrm{se}_w$:

$$
\tilde{g}_w = \bar{g}_w \cdot \max\!\Big(0,\; 1 - \frac{\mathrm{se}_w^2}{\bar{g}_w^2}\Big).
$$

**Search.**
1. Start from the absolute-gradient assignment and from the ranking by $v_b$ at $\mu = 0$; keep the one with lower
   $\hat{J}$.
2. Each round: compute $\mu$ and $\hat{g}$; score all changes by $v_b$; swap the $m$ best upgrades against the $m$
   cheapest downgrades with the budget kept exact; accept if the objective is lower, otherwise halve $m$. Stop when
   $m < 16$.
3. Sweep $p$ from the lowest rate up, starting each rate from the previous rate's assignment.

Units that are swapped: bits for copies, whole words for codes, whole channels for ABFT.

**Annealing arm.** Blocks are sampled with probability proportional to $\exp(v_b / T)$; a worse block is accepted
with probability $\exp(-\Delta J / T)$; $T$ starts at the size of the first rejected step and is multiplied by
$0.9$ per round. It is an arm of Stage 2, not the default.

## 3. Stage 0: implementation and checks (no fault campaign)

Work, in a new branch from `paper1-runP-mean-variance`:
- Per-bit residual rates and storage costs in `netsurf/mean_variance.py`, the MC correction with shared uniform
  numbers, shrinkage, the annealing arm, and the Stage 1 runner.
- Deterministic tests:
  - every $\rho_s(p)$ against exhaustive enumeration of the copies or of a short code word;
  - the storage budget is preserved exactly by every move;
  - with ideal suppression and the correction off, the search returns Run P's `mv_single` mask bit for bit on one
    replication of the pinned checkpoint;
  - two runs with the same seed give identical assignments.
- A timing smoke of the Stage 1 runner: one replication of the pinned checkpoint, two rates, written outside
  `workspace/runs/`.
- The addendum for Stage 2: $K$, the refresh interval, the round limits and the measured build time per mask,
  chosen on the held-out training images 28000-28511 only.

Resources: one process, under 30 minutes of CPU, under 2 GB of memory, no fault outcome on evaluation images
except the timing smoke. Passing all tests is the gate for Stage 1.

## 4. Stage 1: does the formula predict the switch? (frozen)

**Estimand.** For each checkpoint and budget, the bit-error probability at which the mean accuracy of the
positive-gradient mask stops exceeding that of the absolute-gradient mask, both built from 1,024 calibration images,
under ideal suppression.

**Cohort.** The nine ResNet-8 checkpoints of Run Q with the same replications: pinned 1 to 9, the eight others 0 to
7, and replication 4 of retrained seed 7102 excluded (amendment Q-A1). 72 jobs. Each replication evaluates its own
window of 1,000 test images (offset $1000 \times$ replication), as in Runs K, P and Q. These images were used
before.

**Masks.** `abs1024` and `pos1024` at 6.25%, 12.5%, 18.75% and 25%, read from the Run Q `masks.npz` files. Their
hashes are in `publications/paper1_review/data/tecs_runQ/mask_stability.json` (sha256 `e2acaf29…`). No mask is built.

**Fault regime.** Exact INT8 stored-bit XOR faults on kernels, independent with probability $p$, ideal suppression.
Twelve rates: $3\times10^{-4}$, $1\times10^{-3}$, $1.5\times10^{-3}$, $2\times10^{-3}$, $2.5\times10^{-3}$,
$3\times10^{-3}$, $4\times10^{-3}$, $5\times10^{-3}$, $7\times10^{-3}$, $1\times10^{-2}$, $1.5\times10^{-2}$,
$3\times10^{-2}$.

**Fields.** Eight per replication. Field $f$ of replication $r$ of checkpoint $c$ (index in the order of Run Q)
draws one uniform number per bit from `numpy.random.default_rng([20261115, c, r, f])`; a bit is proposed at rate
$p$ when its number is below $p$. Fields are therefore nested across rates and shared by both masks and all
budgets. Each field is saved as its seed and the SHA-256 of the proposed flips at each rate.

**Engine check.** Before any new evaluation, each job replays one saved Run Q schedule with `abs1024` at 12.5% and
must reproduce the saved accuracy exactly, or it stops.

**Predictions, fixed now.** $p^*(\text{checkpoint}, \text{budget})$ is read from
`workspace/runs/paper1_runPQ_crossover_theory_v1_analysis/summary_maxp0.03.json` (sha256 `851cae16dd1b2520…`),
calibration size 1,024. Definition: on the 17 grid rates up to $3\times10^{-2}$, the rate above which the positive
mask has the higher predicted loss at every larger rate, by log-linear interpolation. Values at 12.5%:

| Checkpoint | $p^*$ | Checkpoint | $p^*$ |
|---|---|---|---|
| Pinned | $3.50\times10^{-3}$ | Retrained 7101 | $3.15\times10^{-3}$ |
| Under-trained | $6.86\times10^{-3}$ | Retrained 7102 | $3.36\times10^{-3}$ |
| Pruned 70% | $3.23\times10^{-3}$ | Retrained 7103 | $3.25\times10^{-3}$ |
| Pruned 50% | $4.76\times10^{-3}$ | Retrained 7104 | $3.23\times10^{-3}$ |
| | | Retrained 7105 | $2.58\times10^{-3}$ |

Over the four budgets: $2.4\times10^{-3}$ to $5.2\times10^{-3}$ for the pinned and retrained checkpoints,
$6.9\times10^{-3}$ to $8.8\times10^{-3}$ for the under-trained one, $3.1\times10^{-3}$ to $6.0\times10^{-3}$ for the
pruned ones.

**Analysis.**
- Cell: checkpoint $\times$ budget $\times$ rate. A cell is usable if the mean accuracy of `abs1024` or `pos1024` in
  it is within 0.05 of the clean accuracy of the same windows, measured in the job.
- Difference: `pos1024` minus `abs1024`, paired on the field. Test: pooled-field sign-flip, 100,000 assignments,
  seed 20261116. Holm correction over all usable cells. Resolved means adjusted $p_{\text{Holm}} < 0.05$. Student-t
  intervals over replications are reported alongside.
- A cell is near the switch if its rate lies in $[p^*/1.25,\; 1.25\,p^*]$. Elsewhere the prediction is "positive
  ahead" for $p < p^*$ and "positive behind" for $p > p^*$.
- Measured switch $p^*_{\text{meas}}$ of a checkpoint and budget: log-linear interpolation of the mean difference
  through zero between the two adjacent rates where it changes sign for the last time on the usable rates.

**Decision rules.**
- R1a (sign): no resolved usable cell outside the near zone has the sign opposite to the prediction.
- R1b (coverage): at least 80% of all usable cells outside the near zone have the predicted sign.
- R1c (location): $\log\big(p^*_{\text{meas}} / p^*\big)$ is averaged over budgets within a checkpoint; the 95%
  Student-t interval of the nine checkpoint means, mapped back to a ratio, lies inside $[1/1.5,\; 1.5]$.
- The prediction is confirmed only if R1a, R1b and R1c all hold. The geometric mean of
  $p^*_{\text{meas}} / p^*$ is reported as the bias of the formula whatever the verdict.
- Secondary, descriptive: the measured switch of the under-trained checkpoint is above that of every well-trained
  one; results with 6.25% to 25% shown separately.

**Cost.**

| Item | Value |
|---|---|
| Evaluations | $72 \times 12 \times 8 \times 4 \times 2 = 55{,}296$ (jobs, rates, fields, budgets, masks), plus 72 engine checks |
| Time per evaluation | about 4 s (Run Q) |
| Parallelism | 26 jobs, 2 TensorFlow threads each (52 of 80 cores) |
| Wall time | about 3 hours |
| Memory | about 1.5 GB per job |
| Storage | under 300 MB in `workspace/runs/paper1_runR_stage1_crossover_v1/`, write-once |
| Interpreter | `/projects/kona-assets/manuelbv/.venv-mlperf/bin/python` (TensorFlow 2.14.1 reference kernels) |

## 5. Stage 2: the rate-aware mask under ideal suppression (rules frozen, constants in the addendum)

**Arms**, all at the four budgets, masks from 1,024 calibration images unless stated:
- static: absolute gradient (64 images, the paper's mask), absolute gradient, positive gradient, squared effect,
  FKeras, significance-first;
- `rate_aware_mask` built for the tested rate: analytical; with the MC correction; with the MC correction and
  annealing.

**Grid.** The eight rates $10^{-4}$, $3\times10^{-4}$, $10^{-3}$, $2\times10^{-3}$, $3\times10^{-3}$,
$5\times10^{-3}$, $10^{-2}$, $3\times10^{-2}$. Fresh nested fields, eight per replication, seeds
`[20261117, c, r, f]`.
- All nine checkpoints: masks tested at the rate they were built for.
- Pinned and under-trained: every built-for rate tested at every rate ($8 \times 8$).

**Statistic.** In each usable cell, the gap of an arm is the mean accuracy of the best arm in that cell minus its
own. The headline number of an arm is its largest gap over all usable cells of a checkpoint. Because the best arm is
chosen on the same fields, gaps are biased slightly upward for every arm.

**Decision rules.**
- R2a (never far behind): on no checkpoint is any other arm ahead of the analytical `rate_aware_mask` by 0.5 point
  or more in a usable cell after Holm correction over that checkpoint's usable cells and arms.
- R2b (the static rankings are not): the same test, applied to the absolute gradient and to the positive gradient,
  fails on at least seven of the nine checkpoints for each.
- R2c (corrections): the MC correction, and separately annealing, is kept in the method only if it is resolved ahead
  of the analytical version in at least one usable cell on at least three checkpoints and resolved behind in none.
  Otherwise it is reported and dropped.
- Misjudged rate (descriptive): the gap of a mask built for rate $p_a$ and tested at rate $p_b$, as an $8 \times 8$
  table.

The claim allowed if R2a and R2b hold: on ResNet-8, the rate-aware mask is the only tested method that is never
more than half a point behind the best one at any rate where the model is usable. Not allowed: "better
everywhere", or any statement about other models.

## 6. Stage 3: real protection schemes (rules frozen, constants in the addendum)

**3a. Residual curves, no network.** For 3 copies, 5 copies, SEC-DED over 8 and 64 data bits, and ABFT with channel
sizes of the ResNet-8 layers: simulate $10^7$ protected bits (or as many words) per rate at the twelve Stage 1 rates
and compare the measured residual error rate with $\rho_s(p)$. Rule R3a: every formula lies inside the 99% binomial
interval of its simulation at every rate. ABFT has no assumed formula; its measured curve is reported.

**3b. Accuracy at equal extra storage,** 12.5% and 25%, with every copy and check bit flipping:
- absolute-gradient ranking with 3 copies (the paper's Section 7 arm);
- `rate_aware_mask` with $r_b = \rho_{s(b)}(p)$, 3 copies;
- `rate_aware_mask` choosing none, 3 or 5 copies per bit;
- SEC-DED over 8 and over 64 bits; ABFT.

Pinned checkpoint first, test images 1000 to 1999 as in Section 7, sixteen fresh fields per rate; the nine
checkpoints only if the pinned result justifies it.

Rule R3b: the per-bit choice of scheme is reported as helping only if it is resolved ahead of the absolute-gradient
3-copy arm by 0.5 point or more in a usable cell and resolved behind in none. The comparison with the codes is
descriptive, as in the paper.

## 7. Stage 4: other models (outline, needs its own plan)

The crossover figure for another model needs a differentiable version of it and positive-gradient masks from 512
or more inputs. Order proposed: DS-CNN (the differentiable version must be built), DistilBERT (differentiable
already), then an estimate for Pythia-160M and Qwen3-0.6B.

## 8. Not covered, and claims that will not be made

- Hardware cost: area, power, latency, voters, decoders, scrubbing. Storage bits only.
- Physical fault rates. $p$ is a logical bit-error probability.
- Correlated or multi-bit upsets, and faults in voters or decoders.
- Whole-network optimality: the search is a descent from two starts.
- Any model other than ResNet-8 before Stage 4.
- "MC-guided selection is superior" unless R2c holds.

## 9. What goes into the paper

- R1 holds: the crossover figure and the two-term explanation enter the main text as a confirmed prediction on
  ResNet-8. R1 fails: reported as failed in the record; the explanation stays out or is marked exploratory.
- R2a and R2b hold: the rate-aware mask enters as a method with the claim in Section 5. Otherwise the paper keeps
  the v14 claims.
- R3a holds: the residual curves are added to Section 7 to explain its measured ordering.

## 10. Outputs

- Per stage: a write-once run directory under `workspace/runs/`, an analysis snapshot with its SHA-256, and a results
  record in this folder.
- From the method: the assignment files with hashes; the predicted loss curve of every candidate against $p$; the
  predicted switch rates; the built-for against tested-at table; a map of the share of bits under each scheme per
  layer and bit position.
