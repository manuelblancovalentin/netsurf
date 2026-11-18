# M9.7-A: FKeras reproduction and numerical gate

Implementation worktree: `<repo>/workspace/implementation-worktrees/m97`,
branch `m97-fkeras-adapters`, based on canonical `4d646a83014bd7a0aed11045bf9f3456e341ee56`.
No merge, push, shared-asset edit, training, or publication campaign. All runs here
are engineering checks using already exposed examples. Historical runs stay in
their original worktrees. See `M97_BENCHMARK_ADAPTER_STATUS.md` for the portfolio.

## Source correspondence, not an unqualified reproduction claim

Primary sources: [FKeras paper, Eq. (3), Sections 3.1–3.2](https://kastner.ucsd.edu/wp-content/uploads/2024/05/admin/jats24-fkeras.pdf),
[DOI](https://doi.org/10.1145/3665334), and
[pinned source](https://github.com/KastnerRG/fkeras/blob/68e775c77670c9a1aab4e1b5295b523347d97013/fkeras/metrics/hessian.py).
Repository pin: `68e775c77670c9a1aab4e1b5295b523347d97013`;
`hessian.py` SHA-256 `3e5b61b8e1098167cfd36cef15e22d75c99cef52845de16d825679f3485bdf4e`;
paper PDF SHA-256 `bb327b55efe6c30f5e179c8b90cd3c1772ffd01387e30f050628bc6bc48223d8`.
These are source snapshots, not newly installed dependencies.

| Component | M97 treatment |
|---|---|
| Printed parameter expression | Preserved as raw signed `sum_i λ_i (v_iᵀθ) v_i` |
| `fkeras_paper` ordering score | Magnitude of that sum, separately identified; not signed finite-bit damage |
| Pinned `do_sum_hessian_rank` | `sum_i abs(λ_i² (v_iᵀθ) v_i)`; NOT `abs(sum_i λ_i² (v_iᵀθ) v_i)` |
| Bit ordering | Strict sign/MSB-to-LSB tiers, then descending parameter score, then canonical identity |
| Model/data/storage | Architecture adaptation to pinned DS-CNN, kernel-only affine INT8, not original QKeras experiments |
| Numerical solver | Residual-checked symmetric Lanczos, not reproduction of the source power-iteration defects |
| Fault experiment | Existing persistent Bernoulli fields and ideal suppression; not the paper's original experimental fault laws |

The source first scales an eigenvector by its eigenvalue, then uses that scaled
vector in both its dot product with parameters and its outer coefficient.
It also applies absolute value **before** accumulating components. For one
nonzero eigenpair the two magnitude rankings are proportional. For multiple
eigenpairs, curvature weighting and cancellation differ and rankings can change.
A synthetic rank-reversal test and direct execution of the pinned method's AST
(without importing QKeras) verify this algebra. Neither variant is selected on
the basis of its measured effectiveness. The paper's signed expression is saved;
the magnitude ordering convention is explicit rather than silently asserted to
be an additional printed equation.

The source eigensolver (`top_k_eigenvalues`, lines 287 onward) uses per-tensor
deflation instead of one concatenated-parameter inner product, normalization
with an added epsilon, convergence of successive Rayleigh values rather than a
residual gate, and training-mode HVPs. Its returned vector/value can correspond
to different iteration steps. These are reasons not to promote its numerical
outputs as verified eigenpairs. M97 reproduces the **effective sum score given
validated eigenpairs**, not every historical numerical artifact. Upstream's
`max` strategy is not implemented or mislabeled as this `sum` variant.

## What Hessian is being calculated?

The M95 HVP path differentiates the requantization straight-through companion.
That is a surrogate derivative operator, not automatically the classical Hessian
of a continuously differentiable forward loss. Historical evidence is unchanged.
M97 adds a separately named `MlperfKwsContinuousCompanion` with float64 decoded
kernel variables, fixed decoded INT32 biases, the DS-CNN topology, continuous
ReLU/pooling operations, no activation requantization/clipping, and mean logit
sparse cross-entropy. It is `mlperf-kws-continuous-float64/v1` with loss version
`continuous-logit-sparse-crossentropy/v1`.

This is neither exact integer TFLite inference nor the existing S4–S6 STE forward
model. Exact clean, faulted, isolated-bit and protection outcomes still use
TF 2.14.1 `BUILTIN_REF`, one thread and no delegates. S4–S6 retain their existing
definitions and companion. No Hessian of the discontinuous integer program is
claimed. ReLU boundaries limit finite-difference neighborhoods.

`ContinuousKwsHessian` records every tensor name, TFLite tensor identity,
shape, offset, affine metadata and exact decoded variable correspondence.
Kernel word order is canonical parameter name then C-order flat index; biases,
scales, zero points, and activations are not faultable. The 22,016 kernel words
give 176,128 bits. Bit 7 is the two's-complement sign bit, bit 0 the LSB.
Allocated zero words remain eligible. Affine scales enter decoded θ and loss
derivatives, but no extra bit-delta or scale multiplier is inserted into FKeras.
Its strict bit tiers ignore cross-tensor scale differences by design.

## Eigensolver validation and cost

`leading_hessian_eigenpairs` uses deterministic seeded initialization, a symmetry
probe, signed eigenvalues ordered by descending magnitude (LM), and canonical
vector signs. Negative eigenvalues are retained. A dimension-at-most-64 dense
reference uses `numpy.linalg.eigh`; larger problems use SciPy symmetric ARPACK.
Full-rank large dense Hessians are rejected. Work is bounded by `max_iterations`
and an independent `max_products` HVP cap. Failure to converge, asymmetry,
nonfinite products, excessive residuals or nonorthogonality reject the ranking.

Saved diagnostics include `||Hv−λv||`, the normalized residual
`||Hv−λv|| / max(||Hv||+|λ|, 1e-30)`, orthogonality error, HVP count, solver,
ordering, source pin and numerical mapping. Flat eigenvectors and raw parameter
scores are also serialized, not just their hashes. ARPACK does not expose its
iteration count, so that field is empty rather than invented. Eigenvector sign
changes leave both scores invariant. Repeated eigenspaces cut by k need not
define a unique score across platforms; a fixed seed is not a proof of universal
cross-BLAS bitwise determinism.

Tests include an analytic small nonlinear network, dense-Hessian reference,
finite-difference gradients/HVPs, negative/zero/repeated eigenvalues, work caps,
non-symmetric rejection, sign invariance, repository score correspondence,
mask cardinalities, full-tier equivalence, and schema compatibility.

Bounded KWS calibration (two train examples, k=2, seed 970201) measured:

| Quantity | Value |
|---|---:|
| Eigenvalues, descending absolute value | 5341.90311162786; 434.7146223766696 |
| HVP calls, including validation | 25 |
| Normalized residuals | 3.27e-16; 3.04e-12 |
| Orthogonality error | 6.87e-16 |
| Symmetry probe error | 1.97e-18 |

For a seeded unit direction, central finite differences of the gradient had
relative errors 3.05, 30.47, 304.70, 0.0694, 0.385, 9.89e-7 and 6.44e-6 at
steps 1e-4 through 1e-10 respectively. Coarser steps cross ReLU boundaries;
the two smallest steps pass the unchanged 1e-5 numerical test. All steps remain
reported. This does **not** validate a Taylor approximation for sign-bit faults.

Lanczos costs approximately HVP-count × calibration-forward/backward cost,
plus Krylov-vector algebra. Storage is O(n × ncv) plus the TensorFlow graph and
activations, not a dense 22,016² matrix (~3.88 GB float64). Here ncv=20 is about
3.52 MB of numeric basis storage; graph/process memory is additional. S5/S6 cost
scales with per-example derivatives. Two FKeras formulas share one eigensolve
when seed/options match; their repeated runtime rows must not be added together.
Scores are coordinate-dependent sensitivities, not calibrated loss damage.

## Schema 3.5 and matched engineering configurations

Use existing schema-3 fields, adding the following estimator entries:

```toml
schema_version = "3.5.0"
# existing experiment, benchmark, samples, candidates, fault, execution etc.
[[estimators]]
name = "fkeras_paper" # or fkeras_repository
implementation_version = "fkeras-paper-continuous-kws/v1"
seed = 970201
hessian = {eigenpairs = 2, max_iterations = 100, max_products = 300, tolerance = 0.00001, ordering = "largest_magnitude_signed"}
[[policies]]
name = "fkeras_paper"
estimator = "fkeras_paper"
priority = ["bit_significance_desc", "estimator_score_desc", "stable_identity"]
prespecified = true
```

Repository version: `fkeras-repository-continuous-kws/v1`. Unknown Hessian fields,
other ordering conventions, unsupported architectures, old schema FKeras requests,
or a global/non-tier-first FKeras policy are explicit validation errors. Existing
3.0–3.4 readers and their serialized estimator fields are unchanged. Manifest and
summary versions remain 1.0.0 with additive schema-3.5 fields. No schema-6 campaign
engine was changed.

Maintained bounded examples:

- `config/examples/m97_kws_fkeras_empirical_smoke.toml`: 16 uniformly sampled
  bits, common identities for six methods; four validation clips.
- `config/examples/m97_kws_fkeras_protection_smoke.toml`: six policies, BER 1e-4,
  coverages 6.25%, 12.5%, 18.75%, 25%, two paired fields, four test clips, clean,
  zero- and full-protection controls. Two training clips supply ranking data.

At 12.5% the 22,016 sign bits, and at 25% those plus the 22,016 bit-6 bits,
are selected by all three strict-tier policies. `summary.mask_equivalence`
compares actual saved sets at each budget; report.md exposes identical masks.
Partial tiers may differ. Identical masks with identical proposed fields must
give identical results; separate method labels do not create independent data.

The selected engineering clips were already exposed in earlier development.
They are not newly untouched confirmation inputs. Calibration/validation/test
source partitions are disjoint. Empirical correlations concern only these 16
uniform candidates; noisy small-sample rank agreement is not whole-network
top-k performance. Protection measurements concern two fault fields, not eight
independent hardware observations. Four correct clean clips provide no power
for estimator-superiority claims. See the final evidence receipts below.

## Scope relative to PrioriFI and manuscript citations

[PrioriFI](https://doi.org/10.1145/3779212.3790204),
[author PDF](https://kastner.ucsd.edu/wp-content/uploads/2026/05/admin/asplos2026-priorifi.pdf),
uses Hessian initialization followed by fault-outcome-guided prioritization.
Its Sections 2–3 explicitly distinguish static bit-tier ordering from dynamic
FI and discuss exceptions to bit-level monotonicity. M97 does not implement
that dynamic algorithm or claim its results. A local weight Hessian contains
cross-parameter curvature; FKeras's projected parameter score is not conditioned
on a protection mask, BER or sampled background field. Thus neither “Hessians
contain no interactions” nor “FKeras estimates conditional protection benefit”
is justified.

No existing BibTeX/LaTeX manuscript was found in this checkout. Verified DOI
metadata is in `docs/references.bib` (`weng2024fkeras`, `weng2026priorifi`). Suggested
manuscript locations: related-work paragraph on sensitivity-guided FI; static
baseline mathematical definition and bit-tier policy; implementation-discrepancy
appendix; and scope discussion contrasting static allocation with dynamic FI.
No manuscript text was edited. Zotero was unavailable locally; no profile or
service was created.

## Kona commands (no installation)

```bash
cd <repo>/workspace/implementation-worktrees/m97
<assets>/.venv-mlperf/bin/python -m netsurf validate config/examples/m97_kws_fkeras_protection_smoke.toml
<assets>/.venv-mlperf/bin/python -m netsurf run config/examples/m97_kws_fkeras_protection_smoke.toml
<assets>/.venv-mlperf/bin/python -m netsurf run config/examples/m97_kws_fkeras_empirical_smoke.toml
<netsurf-env>/bin/python -m pytest -q
<assets>/.venv-mlperf/bin/python -m pytest -q tests/test_m97_fkeras.py tests/test_m97_capabilities.py tests/test_m97_integration.py
```

The pinned environment has no Matplotlib. Examples set `plots=false` and produce
immutable reports. Generate figures offline with the already installed canonical
environment, into a **new** derived destination:

```bash
<netsurf-env>/bin/python -m netsurf report RUN_DIRECTORY --output workspace/runs/m97_derived_unique --plots
```

Replace `RUN_DIRECTORY` with the CLI-returned absolute directory. No TensorFlow
inference reruns and no historical file is edited. Do not install a package or
change the reference backend to solve this optional plotting limitation.

The optional upstream-function tests use local source-only checkouts under
`workspace/references/`. Without them, those two source-correspondence tests
skip explicitly; analytic and storage tests still run. To obtain only sources:

```bash
git clone https://github.com/KastnerRG/fkeras workspace/references/fkeras
git -C workspace/references/fkeras checkout --detach 68e775c77670c9a1aab4e1b5295b523347d97013
git clone --filter=blob:none --no-checkout https://github.com/mlcommons/tiny workspace/references/mlcommons-tiny
```

The independent feature test uses `git show 4addd0fa08d216e20637637874e084895f289da4:...`;
it does not check out or download model/dataset collections. Existing canonical
MNIST preparation tests also need the historical workspace checkpoint. This
worktree has a workspace-only symlink to the unchanged canonical checkpoint,
SHA-256 `1c58daf1bb23d1cdbaf38e68e4d957145be62435fd379bc3716904d57aa77780`.

## Evidence receipts

See `M97_VALIDATION_RECEIPTS.md` for finalized run hashes, test counts, measured
mask differences, empirical outcomes, figures and export instructions. Those
receipts distinguish implementation validation from publication evidence.
