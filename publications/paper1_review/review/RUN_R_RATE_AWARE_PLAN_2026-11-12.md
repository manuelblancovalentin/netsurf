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

Amendment R-A1 (12 November 2026, before any Run R outcome, at the researcher's request): ABFT gets a closed-form
residual curve, marked with an asterisk as an approximation (Section 2), and it is now covered by rule R3a
(Section 6). Nothing else changes.

Amendment R-A2 (12 November 2026, before any Run R outcome, at the researcher's request):
- Every symbol is defined in Section 2.0. Three letters that had two meanings were renamed: the assignment is now
  $A$ (was $s$, which also named a scheme); the second bit of a pair is now $b'$ (was $c$, which also named a storage
  cost); the code length is now $(n, d)$ (was $(n, k)$, with $k$ also the field index).
- ABFT wrong repairs and checksum faults are modelled (Section 2.2), from an exploratory count on the pinned
  ResNet-8 (`workflows/abft_pair_count_explore.py`, branch `paper1-runP-mean-variance`, commit a3ce54d). Stage 0
  recomputes these counts under test; Stage 3a checks them by simulation.
- No other design choice, number or rule changes.

Amendment R-A3 (12 November 2026, before any Run R outcome, at the researcher's request; Stage 0 checks still
running, no Stage 1 field drawn):
- **Wider budget range in Stage 1.** Eight budgets: 1%, 3.125%, 6.25%, 12.5%, 18.75%, 25%, 37.5% and 50% of the
  kernel bits. The predicted switch moves with the budget, so a wider range is a sharper test. If the new budgets
  add nothing, later stages may return to the four of the paper.
- **Masks.** `abs1024` and `pos1024` are rebuilt at all eight budgets from the same 1,024 calibration images with
  Run Q's scoring code. At the four old budgets they must equal the Run Q masks bit for bit, or Stage 1 stops.
  Budget sizes are the fraction times 618,880, rounded to the nearest integer (1%: 6,189 bits).
- **Predictions for the new budgets** are computed with the same script and definition as the frozen ones
  (`runPQ_crossover_theory.py`, held-out images 28000-28511, rates up to $3\times10^{-2}$) and frozen by file hash in
  a dated note before any Stage 1 field is drawn. The four old budgets keep their frozen values.
- **Predictions at a grid end.** If $p^*$ lies at the lowest or highest predicted rate ($10^{-4}$ or $3\times10^{-2}$)
  for a checkpoint and budget, that pair enters R1a and R1b only and is left out of R1c.
- **Parallel jobs.** Every stage is split into independent jobs run at the same time. Stage 1 runs one job per
  checkpoint, replication and field (576 jobs), single-threaded, up to 76 at a time. Mask rebuilding and
  predictions run one job per checkpoint and replication (72 jobs each).
- Cost of Stage 1 updated in Section 4. Nothing else changes.

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

### 2.0 Symbols

**Model and data**

| Symbol | Meaning |
|---|---|
| $x$ | One input of the network: a calibration image for ResNet-8. $\mathbb{E}_x$ is the average over the calibration images (512 or 1,024 training images), which stands in for the input distribution. |
| $\theta$ | The clean weights of the model: the stored INT8 codes times their scales. |
| $w$ | Index of one weight (one INT8 kernel entry). ResNet-8 has 77,360 of them. |
| $L_x(\theta)$ | Cross-entropy loss of input $x$ with weights $\theta$. |
| $L(\theta)$ | Average loss over the calibration images, $L(\theta) = \mathbb{E}_x[L_x(\theta)]$. |
| $g_{x,w}$ | Gradient of $L_x$ with respect to weight $w$, at $\theta$. |
| $\bar{g}_w$ | Mean gradient over the calibration images, $\bar{g}_w = \mathbb{E}_x[g_{x,w}]$. |
| $\mathrm{se}_w$ | Standard error of $\bar{g}_w$ over the calibration images. |
| $\tilde{g}_w$ | $\bar{g}_w$ after shrinkage toward zero (Section 2.3). |
| $F_w$ | Curvature of the loss along weight $w$, estimated by the mean squared gradient, $F_w = \mathbb{E}_x[g_{x,w}^2]$. |

**Bits and faults**

| Symbol | Meaning |
|---|---|
| $p$ | Bit-error probability: the probability that any one stored bit, including copies and check bits, flips. A logical rate, not a physical upset rate. |
| $b,\ b'$ | Indices of stored data bits; $b'$ is a second bit when a formula needs a pair. ResNet-8 has 618,880 data bits. |
| $w(b)$ | The weight that bit $b$ belongs to. Each weight has 8 bits. |
| $\Delta_b$ | Change of weight $w(b)$ when bit $b$ alone flips: $\pm 2^j$ times the weight's scale for bit position $j$, with the sign set by the stored bit (the sign bit gives $\mp 128$). Known from the stored code. |
| $e_{x,b}$ | First-order loss change of input $x$ when bit $b$ flips, $e_{x,b} = g_{x,w(b)}\,\Delta_b$. |

**Protection**

| Symbol | Meaning |
|---|---|
| $s,\ s'$ | A protection scheme, such as none, 3 copies, 5 copies, SEC-DED or ABFT. $s'$ is the scheme a bit is moved to. |
| $A$ | An assignment: the scheme given to every data bit. $A(b)$ is the scheme of bit $b$. Choosing $A$ is the problem. |
| $\rho_s(p)$ | Residual flip probability of a data bit under scheme $s$: the probability that the bit is still wrong after voting or correction, when every stored bit flips with probability $p$. |
| $r_b$ | Residual flip probability of bit $b$ under the assignment, $r_b = \rho_{A(b)}(p)$. |
| $\mathrm{cost}_s$ | Extra storage bits per data bit under scheme $s$: 0 for none, 2 for 3 copies, 4 for 5 copies, $(n-d)/d$ for SEC-DED. |
| $S$ | Storage budget: the total number of extra storage bits allowed. 12.5% means $S = 0.125 \times 618{,}880$. |
| $n,\ d$ | SEC-DED code word: $n$ stored bits of which $d$ are data bits. |
| $G$ | Number of weights in one output channel of a kernel (one ABFT group). |

**Random fault field**

| Symbol | Meaning |
|---|---|
| $\xi_b$ | 1 if bit $b$ ends up flipped after protection, 0 otherwise; $\xi_b \sim \mathrm{Bernoulli}(r_b)$, independent across bits for none and copies. |
| $\mathbb{E}_\xi$ | Average over the random flips $\xi$ (over fault fields). |
| $X_w$ | Total displacement of weight $w$ in one fault field, $X_w = \sum_{b:\,w(b)=w} \xi_b\,\Delta_b$. $X$ is the vector of all $X_w$. |
| $\mu_w$ | Average displacement of weight $w$ over fault fields, $\mu_w = \mathbb{E}_\xi[X_w]$. |
| $V_w$ | Variance of the displacement of weight $w$ over fault fields. |
| $\mathrm{Cov}(\xi_b,\xi_{b'})$ | Covariance of the flips of two bits. Zero for none and copies; positive within a code word or an ABFT channel, whose failures come together. |

**Objective and search**

| Symbol | Meaning |
|---|---|
| $J(A;p)$ | Exact expected loss of the faulty model under assignment $A$ at rate $p$. |
| $\hat{J}(A;p)$ | The mean-and-variance approximation of $J$ (Section 2.2). |
| $K$ | Number of fault fields used by the MC correction. |
| $k$ | Index of one of those fields, $k = 1,\dots,K$; $X_k$ is its displacement vector. |
| $u_{b,k}$ | Uniform random number in $[0,1)$ drawn once for bit $b$ in field $k$; $\xi_{b,k} = \mathbb{1}[u_{b,k} < r_b]$. |
| $\hat{C}$ | MC estimate of what $\hat{J}$ misses. |
| $\hat{g}_w$ | Gradient used to score bits: at the average faulty weights $\theta+\mu$, or averaged over the $K$ faulted models. |
| $v_b$ | Value of moving bit $b$ from scheme $s$ to $s'$: predicted drop of $\hat{J}$ per extra storage bit. |
| $m$ | Number of bits swapped in one round of the search. |
| $T$ | Temperature of the annealing arm: the size of a loss increase that is still accepted fairly often. |
| $\Delta J$ | Change of the objective caused by one proposed swap. |

**Analysis (Stages 1 and 2)**

| Symbol | Meaning |
|---|---|
| $p^*$ | Predicted switch rate: the rate above which the positive-gradient mask has the higher predicted loss. |
| $p^*_{\text{meas}}$ | Measured switch rate: where the measured accuracy difference crosses zero. |
| $p_a,\ p_b$ | In Stage 2, the rate a mask is built for ($p_a$) and the rate it is tested at ($p_b$). |
| $\mathrm{ckpt},\ \mathrm{rep},\ \mathrm{field}$ | Integers in the random seeds: checkpoint index in the order of Run Q (0 to 8), replication, and field number. |

### 2.1 Protection schemes

Voters and decoders are ideal unless stated.

| Scheme | $\rho_s(p)$ | $\mathrm{cost}_s$ |
|---|---|---|
| None | $p$ | 0 |
| Ideal suppression (the paper's masks) | $0$ | counted as 1 bit per bit, as in the paper |
| 3 copies | $3p^2 - 2p^3$ | 2 |
| 5 copies | $10p^3 - 15p^4 + 6p^5$ | 4 |
| SEC-DED, code word of $n$ bits with $d$ data bits | $p\,[1 - (1-p)^{n-1}]$, double errors left as they are, miscorrection ignored | $(n-d)/d$, whole words |
| ABFT, one faulty weight repaired per output channel of $G$ weights$^*$ | $p\,[1 - (1-p)^{8(G-1)}] \approx 8(G-1)\,p^2$, plus wrong repairs (Section 2.2) | two checksum words per channel |

**Where the formulas come from.**
- *Copies:* the vote is wrong when a majority of copies flip. For 3 copies that is 2 or 3 of them,
  $3p^2(1-p) + p^3 = 3p^2 - 2p^3$; for 5 copies, 3 or more of 5.
- *SEC-DED:* one error per code word is repaired, so a flipped data bit stays wrong when at least one of the other
  $n-1$ stored bits of its word has also flipped.
- *ABFT:* one faulty weight per channel is repaired, so a flipped bit stays wrong when at least one of the
  $8(G-1)$ bits of the other weights of its channel has also flipped.

$^*$ **The ABFT curve is an approximation.** It is exact for the model the paper simulates (Section 7,
`workflows/correction_mechanisms_resnet8.py`): two checksum words per output channel of each kernel locate and
repair one faulty weight; a channel with two or more faulty weights is left as it is; the checksum words never
flip. A real ABFT differs in three ways, each of which makes it worse than the curve:
1. **Wrong repairs.** With two faulty weights in one channel, the checksums can point at a third, healthy weight,
   which is then changed. This occurs at the same order, $p^2$, as the main term. It is modelled in Section 2.2.
2. **Faulty checksums.** The checksum words are stored too and can flip. Also modelled in Section 2.2.
3. **Where it checks.** A real ABFT checks the computed outputs of a layer, not the stored weights. Then where a
   fault seems to be depends on the input, so it can only be estimated by sampling inputs. Run R does not model
   this; applying ABFT to the stored weights, as the paper does, favours it.

The curve differs per layer, because $G$ does (from 16 to 576 weights in the pinned ResNet-8).

### 2.2 Objective

**Exact objective.** For a storage budget $S$ and rate $p$, choose the assignment $A$ that minimizes

$$
J(A;p) = \mathbb{E}_{x}\,\mathbb{E}_{\xi}\big[L_x(\theta + X)\big],
\qquad X_w = \sum_{b:\,w(b)=w} \xi_b\,\Delta_b,
\qquad \xi_b \sim \mathrm{Bernoulli}(r_b),
\qquad r_b = \rho_{A(b)}(p),
$$

subject to

$$
\sum_b \mathrm{cost}_{A(b)} \le S .
$$

In words: the average loss over calibration images and over random fault fields, where each bit flips with the
residual probability of the scheme it was given.

**Mean-and-variance approximation.**

$$
\mu_w = \sum_{b:\,w(b)=w} r_b\,\Delta_b,
\qquad
V_w = \sum_{b:\,w(b)=w} r_b\,(1-r_b)\,\Delta_b^2,
$$

$$
\hat{J}(A;p) = \big[L(\theta+\mu) - L(\theta)\big] + \tfrac12 \sum_w F_w\,V_w .
$$

The first term is the loss at the average faulty weights; the second is the spread of the weights around that
average, weighted by curvature.

**Grouped failures (codes and ABFT).** Flips inside one code word or one ABFT channel are not independent, so the
variance term gains

$$
\sum_{\substack{b \ne b' \\ \text{same group}}} \mathrm{Cov}(\xi_b,\xi_{b'})\;\mathbb{E}_x\big[e_{x,b}\,e_{x,b'}\big],
$$

where the group is the code word or the output channel.

**ABFT wrong repairs and checksum faults.** Decoder assumed: the two checksums are $C_1 = \sum_{i=1}^{G} q_i$ and
$C_2 = \sum_{i=1}^{G} i\,q_i$ over the integer codes $q_i$ of a channel, stored as 32-bit words. With integer errors
$\delta_i$ the decoder computes $\sigma_1 = \sum_i \delta_i$ and $\sigma_2 = \sum_i i\,\delta_i$ (minus any checksum
error). If $\sigma_1 = 0$ it does nothing. If $\ell = \sigma_2/\sigma_1$ is an integer in $1,\dots,G$ it subtracts
$\sigma_1$ from weight $\ell$. Otherwise it leaves the channel as it is.

To order $p^2$, every pair of flips in one channel has probability about $p^2$, and its outcome is fixed by the
stored codes. So the wrong-repair rate of a channel is $p^2$ times a count:

$$
\Pr[\text{wrong repair in a channel}] \approx p^2\,\big(N_{\text{ww}} + N_{\text{wc}}\big),
$$

- $N_{\text{ww}}$: the number of pairs of data bits in different weights of the channel whose syndrome points at a
  healthy weight;
- $N_{\text{wc}}$: the same for pairs of one data bit and one checksum bit.

A wrong repair moves the pointed weight by $-\sigma_1$, so it adds $-p^2\sigma_1$ to that weight's $\mu_w$ and
$p^2\sigma_1^2$ to its $V_w$. A single checksum flip on its own does nothing under this decoder ($\sigma_1 = 0$, or
$\ell = 0$, which is out of range).

Exploratory count on the pinned ResNet-8 (commit a3ce54d), as shares of the pairs of each kind:
- pairs of data bits in different weights of one channel: 19.3% end in a wrong repair, 6.2% cancel in $\sigma_1$
  and go undetected, and the rest are detected and left (15% for $G=16$ up to 19% for $G=576$);
- pairs of one data bit and one checksum bit: 13.6% end in a wrong repair.

These shares depend on the decoder rules above (index from 1, nothing when $\sigma_1 = 0$, 32-bit checksums). Other
rules give other counts, by the same method.

**MC correction.** With $K$ fault fields $X_1,\dots,X_K$:

$$
\hat{C} = \frac1K \sum_{k=1}^{K} \big[L(\theta + X_k) - L(\theta+\mu)\big] - \tfrac12 \sum_w F_w\,V_w .
$$

One uniform number $u_{b,k}$ per stored bit and field is drawn once, and $\xi_{b,k} = \mathbb{1}[u_{b,k} < r_b]$.
The same numbers serve every candidate assignment and every rate, so two candidates differ only through the bits
whose scheme changed.

### 2.3 Search

**Value of moving bit $b$ from scheme $s$ to $s'$, per extra storage bit.**

$$
v_b = \frac{(\rho_s - \rho_{s'})\,\Big[\hat{g}_{w(b)}\,\Delta_b + \tfrac12\,(1 - \rho_s - \rho_{s'})\,F_{w(b)}\,\Delta_b^2\Big]}{\mathrm{cost}_{s'} - \mathrm{cost}_s},
\qquad \rho_s = \rho_s(p).
$$

$\hat{g}_w$ is the gradient at $\theta+\mu$, or the mean signed gradient over the $K$ faulted models,
$\frac1K\sum_k \nabla_w L(\theta + X_k)$, when the MC correction is on.

**Calibration noise.**

$$
\tilde{g}_w = \bar{g}_w \cdot \max\!\Big(0,\; 1 - \frac{\mathrm{se}_w^2}{\bar{g}_w^2}\Big).
$$

Where the sign of the mean gradient is not resolved, the signed term vanishes and the score falls back on
magnitude.

**Steps.**
1. Start from the absolute-gradient assignment and from the ranking by $v_b$ at $\mu = 0$; keep the one with lower
   $\hat{J}$.
2. Each round: compute $\mu$ and $\hat{g}$; score all moves by $v_b$; swap the $m$ best upgrades against the $m$
   cheapest downgrades with the budget kept exact; accept if the objective is lower, otherwise halve $m$. Stop when
   $m < 16$.
3. Sweep $p$ from the lowest rate up, starting each rate from the previous rate's assignment.

Units that are swapped: bits for copies, whole words for codes, whole channels for ABFT.

**Annealing arm.** Blocks of moves are sampled with probability proportional to $\exp(v_b / T)$; a block that
raises the objective by $\Delta J$ is accepted with probability $\exp(-\Delta J / T)$; $T$ starts at the size of the
first rejected step and is multiplied by $0.9$ per round. It is an arm of Stage 2, not the default.

## 3. Stage 0: implementation and checks (no fault campaign)

Work, in a new branch from `paper1-runP-mean-variance`:
- Per-bit residual rates and storage costs in `netsurf/mean_variance.py`, the grouped-failure term, the ABFT pair
  counts, the MC correction with shared uniform numbers, shrinkage, the annealing arm, and the Stage 1 runner.
- Deterministic tests:
  - every $\rho_s(p)$ against exhaustive enumeration of the copies or of a short code word;
  - the ABFT pair counts against a brute-force decoder on small random channels;
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
window of 1,000 test images (offset $1000 \times \mathrm{rep}$), as in Runs K, P and Q. These images were used
before.

**Masks.** (Amendment R-A3: eight budgets, rebuilt; see the top of this plan.) `abs1024` and `pos1024` at 6.25%,
12.5%, 18.75% and 25%, read from the Run Q `masks.npz` files. Their
hashes are in `publications/paper1_review/data/tecs_runQ/mask_stability.json` (sha256 `e2acaf29…`). No mask is built.

**Fault regime.** Exact INT8 stored-bit XOR faults on kernels, independent with probability $p$, ideal suppression.
Twelve rates: $3\times10^{-4}$, $1\times10^{-3}$, $1.5\times10^{-3}$, $2\times10^{-3}$, $2.5\times10^{-3}$,
$3\times10^{-3}$, $4\times10^{-3}$, $5\times10^{-3}$, $7\times10^{-3}$, $1\times10^{-2}$, $1.5\times10^{-2}$,
$3\times10^{-2}$.

**Fields.** Eight per replication. Field $\mathrm{field}$ of replication $\mathrm{rep}$ of checkpoint
$\mathrm{ckpt}$ draws one uniform number per bit from
`numpy.random.default_rng([20261115, ckpt, rep, field])`; a bit is proposed at rate $p$ when its number is below
$p$. Fields are therefore nested across rates and shared by both masks and all budgets. Each field is saved as its
seed and the SHA-256 of the proposed flips at each rate.

**Engine check.** Before any new evaluation, each job replays one saved Run Q schedule with `abs1024` at 12.5% and
must reproduce the saved accuracy exactly, or it stops.

**Predictions, fixed now.** $p^*(\mathrm{ckpt}, \text{budget})$ is read from
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
  seed 20261116. Holm correction over all usable cells. Resolved means Holm-adjusted p-value below 0.05. Student-t
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
| Evaluations | $72 \times 12 \times 8 \times 8 \times 2 = 110{,}592$ (replications, rates, fields, budgets, masks), plus one engine check per job (amendment R-A3; was 55,296 with four budgets) |
| Time per evaluation | about 6.2 s (Stage 0 timing smoke; the plan first said about 4 s) |
| Parallelism | 576 jobs (checkpoint, replication, field), single-threaded, up to 76 at a time |
| Wall time | about 2.5 to 3 hours |
| Memory | about 1.4 GB per job, about 110 GB at 76 jobs (430 GB available) |
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
`[20261117, ckpt, rep, field]`.
- All nine checkpoints: masks tested at the rate they were built for.
- Pinned and under-trained: every built-for rate $p_a$ tested at every rate $p_b$ ($8 \times 8$).

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

**3a. Residual curves, no network.** For 3 copies, 5 copies, SEC-DED over 8 and 64 data bits, and ABFT with the
channel sizes of the ResNet-8 layers: simulate $10^7$ protected bits (or as many words or channels) per rate at the
twelve Stage 1 rates and compare the measured residual error rate with $\rho_s(p)$. For ABFT the simulation uses the
decoder of Section 2.2, with one curve per channel size $G$, and also measures the wrong-repair rate and the share of
wrong repairs caused by checksum flips. Rule R3a: every formula, and the ABFT wrong-repair rate
$p^2(N_{\text{ww}} + N_{\text{wc}})$, lies inside the 99% binomial interval of its simulation at every rate where
the expected count is at least 30. Checks on computed outputs, rather than stored weights, are outside R3a.

**3b. Accuracy at equal extra storage,** 12.5% and 25%, with every copy and check bit flipping:
- absolute-gradient ranking with 3 copies (the paper's Section 7 arm);
- `rate_aware_mask` with $r_b = \rho_{A(b)}(p)$, 3 copies;
- `rate_aware_mask` choosing none, 3 or 5 copies per bit;
- SEC-DED over 8 and over 64 bits; ABFT as in the paper, and ABFT with wrong repairs and checksum flips.

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
- ABFT that checks computed outputs, whose fault location depends on the input.
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

## Freeze note F1 (12 November 2026, 16:25 CST, before any Stage 1 field is drawn)

Amendment R-A3 prerequisites are complete:
- Masks: `workspace/runs/paper1_runR_masks_v1/` (72 replications, read-only), `runR_masks.py` at commit 059af83 of
  branch `paper1-runR-rate-aware`. At the four old budgets every mask equals Run Q's bit for bit (no job failed its
  check).
- Predictions: `workspace/runs/paper1_runR_predictions_v1/predictions.json`, sha256
  `bf38ffd6651bfa179352c18c8205d13465315e4f0019634642da6e0995f7a037`, read-only. Recomputing the four old budgets
  reproduces the frozen values exactly (largest log ratio 0.0); the old budgets keep the frozen values by rule.
- One pair lies at the grid end: under-trained at 50% ($p^* = 3\times10^{-2}$). It enters R1a and R1b only.
- Predicted $p^*$ at 1% / 50%: pinned $4.9\times10^{-3}$ / $2.2\times10^{-2}$; under-trained $1.1\times10^{-2}$ / grid
  end; pruned 70% $3.7\times10^{-3}$ / $8.4\times10^{-3}$; pruned 50% $5.4\times10^{-3}$ / $1.9\times10^{-2}$; retrained
  seeds $1.6\times10^{-3}$ to $4.1\times10^{-3}$ / $1.8\times10^{-2}$ to $2.2\times10^{-2}$.

## Amendment R-A4 (12 November 2026, before any Stage 1 or Stage 2 outcome, at the researcher's request)

The annealing arm is removed from Stage 2. As specified, one temperature sets both the acceptance of a block and the
sampling of single bits, and in Stage 0 the sampling became uniform (Stage 0 record, Section 3). Stage 2 keeps the
analytic `rate_aware_mask` and the version with the MC correction; rule R2c applies to the MC correction only.
A corrected annealing schedule, if wanted later, needs its own amendment. Nothing else changes.

## Analysis note A1 (12 November 2026, Stage 1 running, before any Stage 1 outcome is read)

Two details the Stage 1 analysis needs and the plan did not state:
- A usable cell whose mean difference is exactly zero counts as not agreeing with the prediction in R1b.
- A checkpoint and budget whose mean difference never changes sign on its usable rates has no measured switch. It is
  left out of R1c (as are grid-end predictions), and the number of such pairs is reported. A checkpoint left with no
  pair is left out of the R1c interval, which is then computed over the remaining checkpoints and reported with
  their number.
- Clean accuracy for the usability criterion: the mean over replications of the clean accuracy of each replication's
  window, measured in the Stage 1 jobs.
