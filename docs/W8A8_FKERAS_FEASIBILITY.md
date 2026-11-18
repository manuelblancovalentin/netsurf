# Full-eligible W8A8 FKeras feasibility gate

This is an engineering preflight on branch `codex/w8a8-fkeras-feasibility`,
starting from transformer source commit `9e258cd2390b4d07bff9086cf0611e22da392b31`.
It does not change or launch any six-method campaign. Real-model preflights are
gated on the existing Qwen W8A8 campaign's finalized archive and successful
runtime receipt. Until that condition is met, all three model decisions are
**PENDING**; no FKeras coverage cell or matched comparison is measured.

## Scientific definition

The [FKeras paper, Eq. (3) and Section 3.1](https://kastner.ucsd.edu/wp-content/uploads/2024/05/admin/jats24-fkeras.pdf)
projects the continuous loss Hessian onto its leading eigenpairs,
`H' = sum_i lambda_i (v_i dot theta) v_i`, then orders every stored-word bit
by **bit significance tier first**, using the parameter score only within a
tier. The pinned repository's sum implementation differs algebraically and is
tracked separately in `M97_FKERAS_REPRODUCTION.md`. This gate concerns the
published-form expression and does not substitute a diagonal Hessian,
Fisher score, sampled-parameter HVP, or a global cross-tier score order.

For these derived W8A8 models, `theta` is every eligible QLinear qweight code
decoded as `float32(code) * fixed_per_output_row_scale` in quantization-manifest
order. The companion replaces all eligible QLinear modules by FP32 Linear
modules, freezes noneligible parameters, and uses task cross-entropy for SST-2
or teacher-forced token NLL for WikiText-2. This is a *continuous decoded-weight
surrogate loss*. It omits dynamic activation requantization and is not a
Hessian of the authoritative `torch._int_mm` W8A8 integer executable. Any
eventual protection outcome must still be measured on that integer executable.

The bounded probe uses one already exposed model-selection input: the first
frozen SST-2 train record or a 128-token WikiText-2 validation window beginning
at token 1024. No final-evaluation input enters it. A two-product symmetry
probe precedes a deterministic full-vector power iteration with at most six
more HVP calls. The preflight records each signed Rayleigh
value and normalized `||Hv-lambda v||/(||Hv||+|lambda|)`. The preflight tolerance
is `1e-3`. One converged candidate still does not constitute the paper's top-8
score; a nonconverged probe is reported without a rank or protection claim.
The full top-8 eigenpair solver would require its own residual, symmetry,
orthogonality, stability and resource gates before any matched campaign could
be frozen. No such campaign is frozen in this worktree.

## Exact eligible universes and numeric-memory projection

These counts and digests come from the existing frozen full-eligible-qweight
plans. Non-qweight weights, scales, biases, activations and normalization are
outside the declared faultable universe. Qwen's separately stored output head
is included. The projection assumes eight float64 eigenvectors, a 20-vector
float64 Lanczos basis, FP32 decoded weights and two FP32 HVP vectors, plus a
separate score/mask phase. It excludes model activations, original state,
autograd graphs, allocators and ZIP overhead, so it is **not a RAM ceiling**.

| Model | Eligible modules | INT8 codes | Eligible bits | Universe SHA-256 | Lanczos basis | Numeric peak projection |
|---|---:|---:|---:|---|---:|---:|
| DistilBERT/SST-2 | 38 | 43,058,688 | 344,469,504 | `de623b6098eb8de1fc5934436cb45dbc3022f852d9da1655ada146a75a8e3284` | 6.42 GiB | 9.46 GiB |
| Pythia-160M | 49 | 123,568,128 | 988,545,024 | `4ddd961b504905f004030cb79baa44fee55eb67b500f60544c282490173af6f9` | 18.41 GiB | 27.16 GiB |
| Qwen3-0.6B Base | 197 | 595,984,384 | 4,767,875,072 | `6ece09559a49e6a0379f1844f2fa072ed1992d586cd97375e7d2674416c6de85` | 88.81 GiB | 130.99 GiB |

These numeric projections already exceed the original campaign's 8, 12 and
64 GiB per-process limits, respectively. A future full FKeras campaign would
need a distinct measured resource envelope and matched-plan freeze. The
bounded HVP probe uses no Lanczos basis, score files or masks. Its per-call
wall time and peak RSS will supply a lower-bound `500 * fastest HVP seconds`
estimate for a 500-product construction; that lower bound excludes mask
construction and cannot establish feasibility alone.

At 12.5% coverage every strict-tier method protects all sign bits, and at 25%
it protects the sign and bit-6 tiers. These masks coincide with MSB-first at
those exact budgets regardless of the weight scores. A matched analysis must
avoid counting them as independent policy outcomes. At 6.25% and 18.75%,
within-tier weight scores can change allocation. The `uint64` tier/score key
primitive passes through the existing exact streaming top-budget writer with
global-ordinal tie breaking; it has only synthetic validation so far.

## Gate and bounded command

Use the pinned Kona interpreter. The actual preflight command is intentionally
deferred while the Qwen campaign is active. The destination must be a new
write-once directory in this worktree. An external foreground timeout bounds
one model's execution even if a single backend operator stalls:

```bash
cd <repo>/workspace/implementation-worktrees/w8a8-fkeras-feasibility
timeout --signal=TERM --kill-after=60s 2h \
  <assets>/envs/torch-2.9.1-cpu-py311/bin/python \
  scripts/preflight_w8a8_fkeras_full_v1.py distilbert \
  workspace/runs/w8a8_fkeras_distilbert_full_preflight_v1 \
  --max-products 6 --tolerance 0.001 --max-seconds 7200 --max-rss-gib 128
```

The same command can substitute `pythia` or `qwen` and a distinct destination
after scientific and resource admission. It verifies the target's frozen plan,
READY asset record and full eligible universe; the Qwen finalized archive
gate fails before creating output while Qwen is still running. A failed or
interrupted output is retained and must be investigated, never silently
restarted with a different identity. No new BER points, masks, schedules or
faulted-model inferences are part of this preflight.

## Current bounded validation

- Canonical `netsurf` Python 3.11: five synthetic tests passed; the PyTorch
  fixture is skipped because that environment has no PyTorch.
- Pinned Kona PyTorch interpreter: the no-pytest two-QLinear fixture passed
  exact decoded-weight forward mapping and a second derivative. This
  interpreter does not have pytest; no package was installed.
- The repository-wide suite was attempted in this fresh worktree: 387 passed,
  10 failed and 21 skipped. The ten failures refer to ignored historical
  checkpoint/plan files absent from this worktree's `workspace/`; no active
  run was linked into the new worktree to satisfy them. The source-independent
  FKeras/W8A8 subset passed 20 tests with one PyTorch skip.
- The real-model command failed closed while Qwen was active, before making
  its destination directory. No real-model HVP has run yet.

Until real full-population HVP timings, residuals, memory and multi-eigenpair
projection pass for an individual model, its published-form FKeras feasibility
decision remains **BLOCKED/PENDING EVIDENCE**, and its coverage cell remains
**UNMEASURED**. A head-only, layer-only or sampled HVP would be a separately
named diagnostic, never a full-population FKeras result.
