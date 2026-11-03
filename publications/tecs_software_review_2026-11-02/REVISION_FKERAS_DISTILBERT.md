# Published FKeras and targeted DistilBERT revision

Unified internal-review target: **9 November 2026**. This revision is manuscript
authoring and offline analysis only. It does not change model execution,
estimators, masks, plans, historical bundles or the running research worktrees.

## Methodological decision

The sole main-paper FKeras baseline is the published projection
`sum_j lambda_j (v_j^T theta) v_j`, absolute component ranking within descending
bit-significance tiers. The numerical KWS adaptation uses two signed
magnitude-ordered eigenpairs of a continuous float64 companion, with canonical
ties. Those assumptions are disclosed rather than asserted to be every setting
of the original implementation. The repository-variant series is removed from
main figures and the method table. Historical variants remain archived.
Original Holm families are retained; removing a displayed series is not
permission to reduce multiplicity after seeing outcomes.

The common 1000-clip KWS grid was **not found finalized** in the inspected
canonical/manuscript outputs. Only preflight/plan records were present. The
128-clip study is retained separately, with its cohort distinction stated once.
No observations are spliced into the 1000-clip curves. A supplementary heatmap
uses only this complete measured 3-BER × 4-budget grid, fixed MSB reference and
positive-is-improvement convention. Each cell's original paired interval and
Holm value accompany it in `evidence/fkeras_heatmap_cells.csv`. It is a view of
the same evidence, not an extra replication or the pending common-grid study.

## DistilBERT evidence admitted

Source is pinned in `config/paper1_distilbert_source.json`. Archive location:

`/projects/netsurf/manuelbv/workspace/implementation-worktrees/transformer-integration/workspace/runs/distilbert_sst2_classifier1024_v4_run1_export/`

| Binding | SHA-256 |
|---|---|
| Complete archive | `9df099fa58b125315aa68ec528cb3ee71bbe00c7f7c33744e3f1f274849259a4` |
| Compact archive | `f677ce2cf0f0da5eaa33cbc14aaa1def52cca24055106da55520e51d80a72529` |
| Frozen plan | `735ad6e8531628059bf891075e37885026b220a586bc2d0dc640e507238dc7d7` |
| Original READY snapshot | `124de29e1f918bc3acc0153124dbfdf111f3b79984135a21652f7e89e222fc22` |
| Final analysis scientific payload | `1abf4fe3e5901b432dc37578298e0dfda80367d598169aeaace3990ee3d68a7e` |

The original READY snapshot is inside the archive and matches the plan. The
current mutable global manifest is deliberately not used to redefine that
historical identity. Fourteen archived source files match the frozen source
hashes. The audit verifies all 101 bound run records (three selections, clean,
96 fields, analysis), all 96 independently seeded schedules, 768 policy
evaluations, exact mask cardinalities and proposed/applied/suppressed counts.
All zero-applied states match clean inference, not only full-protection controls.

Fault support is exactly 1,024 FP32 bits in the first 32 `classifier.weight`
scalar words, not the transformer's full inspectable parameter population.
The hash-pinned safetensors header gives shape `[2,768]` (FP32); row-major
flat indices 0–31 are row 0, columns 0–31. The pinned model configuration
maps output row 0 to NEGATIVE and row 1 to POSITIVE, so every targeted scalar
feeds only the NEGATIVE logit. The existing frozen calibration is unchanged;
architecture- and class-stratified testing requires a new protocol.
Ranking uses 32 training examples; final data are validation indices 32–287.
Prior clean-reference evaluation saw all 872 validation examples. Engineering
fault outcomes used indices 0–31; the frozen plan records their exclusion.
This is a fixed-cohort targeted case study, not inference-naive confirmation.

Every BER has 32 fields, with one fixed mask per method/budget. The original
analysis family comprises isolated versus random and isolated versus
significance, two budgets, three BERs: 12 tests. Recomputed 10,000-draw
field-bootstrap intervals and symmetric-null sign-flip p-values reproduce the
archived analysis; the original Holm correction is unchanged. Accuracy
intervals added here are explicitly descriptive and are not a new
confirmatory family. They do not estimate input-population or random-mask
population uncertainty.

All 768 risk/accuracy records were separately verified from stored logits and
hash-checked dataset labels: 196,608 dependent example outcomes. This reads
data and logits; it performs **zero model inferences**. Cross-entropy is capped
at 30 nats per example, including finite extreme logits; nonfinite/failed
outputs receive the same penalty. No failed logits were observed. At BER
16/1024, isolated ranking at either interior budget has 144 capped outcomes
out of 8,192, significance has zero, and unprotected inference has 584.
These outcome counts are not independent statistical samples.

At BER 16/1024 and 128 protected bits:

| Method | Mean capped CE | Mean accuracy |
|---|---:|---:|
| Unprotected | 2.499798 | 0.851685 |
| Random | 2.498730 | 0.851685 |
| FP32 significance | 0.467338 | 0.898560 |
| Exact isolated | 0.925460 | 0.897705 |
| Clean / full | 0.461013 | 0.898438 |

Significance-minus-isolated accuracy is +0.08545 pp, descriptive paired 95%
interval [0, 0.18311] pp. Its loss reduction is 0.45812 nats, original paired
interval [0.20842, 0.73082], original Holm p=0.00120. Isolated-minus-random loss
benefit is 1.57327 nats but original Holm p=0.08279. All other BERs/budgets and
null/adverse effects are retained in the manuscript effect table and CSVs.
No new superiority claim is inferred from the extra benchmark.

Calibration mismatch, limited calibration count, FP32 representation effects
and simultaneous-fault interactions remain confounded. To distinguish them,
one would need prespecified calibration resampling with a fixed fault law and
independently varied background fields with fixed calibration. That experiment
has **not** been performed by this manuscript revision.

## Computation, novelty and unfinished work

Exact isolated scoring used 1,025 requests / 32,800 example evaluations. Final
comparisons used 769 requests / 196,864 example evaluations. Random and
significance scores require no construction inference. A 5.86-hour number in
the frozen plan is a prior forecast, not measured campaign runtime. The
finalized records lack a wall-time timer; ZIP modification times are not a
substitute. The manuscript reports exact work counts and this limitation.

No finalized equal-compute random-search/greedy-swap comparison was located in
the transformer run tree. Implementations or planning elsewhere are not
treated as results. Related work distinguishes static scoring, empirical
search, learned/RL allocation and physical mitigation. An adaptively selected
or learned mask may be fixed at deployment and still exploit interactions.

Pythia BF16-stored/FP32-compute and Qwen3-0.6B-Base native-BF16 now have
READY clean assets and finalized bounded three-bit engineering fault smokes,
but no admitted finalized BER protection-effectiveness evidence. These distinct
readiness levels do not qualify as publication comparisons. The
paper retains the software-only scope and makes no physical-protection claim.
The hostile review still identifies substantial novelty overlap with FCCM,
FKeras, PrioriFI, learned allocation and standard reliability theory. The new
case study does not resolve the existing independent-pool conditional-versus-
isolated uncertainty.

## Reproduce the manuscript revision, not the experiments

From the isolated `paper1-expanded` worktree:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=workspace/cache/matplotlib \
 /mnt/raid0/ml/conda_envs/manuelbv/envs/netsurf/bin/python \
 publications/tecs_software_review_2026-11-02/build_evidence.py --refresh-draft

/mnt/raid0/ml/conda_envs/manuelbv/envs/netsurf/bin/python \
 publications/tecs_software_review_2026-11-02/build_distilbert_evidence.py --refresh-draft

/projects/kona-assets/manuelbv/envs/torch-2.9.1-cpu-py311/bin/python \
 publications/tecs_software_review_2026-11-02/verify_distilbert_logits.py

cd publications/tecs_software_review_2026-11-02/overleaf
latexmk -pdf -interaction=nonstopmode main.tex
```

The logit verification uses the installed PyArrow reader, not PyTorch inference.
The archive-only analysis uses NumPy and standard-library readers. The full
review ZIP includes the original compact/complete source archives and their
checksums, new offline verification receipts, paired CSVs, provenance, tests and
the hostile review. The Overleaf ZIP needs no Python or datasets to compile.
Do not overwrite a previously distributed ZIP; each export is separately named.
