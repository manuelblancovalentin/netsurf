# PrioriFI-derived allocation: cross-benchmark engineering gate

Status: **bounded engineering only; not a measured publication ranking method**.
The original finalized runs, asset locks, and comparative SQLite database were
read, not changed. New receipts are under this worktree's ignored `workspace/runs/`.

## What is and is not reproduced

The pinned public [PrioriFI smart-pixel example](https://github.com/KastnerRG/priorifi/blob/6e7637b92315e5df75bb6ce258972bdf1131bc20/examples/smart-pixel/fault_injection_campaign.py)
uses repository-form FKeras Hessian parameter scores to order words inside
significance-tier probe queues. Its feedback selects which tier to probe next;
its default feedback is changed predictions among initially correct inputs.
It does **not** specify a fixed-budget protection allocator. Completing every
probe gives an empirical isolated-fault inventory, not a new protection rule.

`netsurf.priorifi` preserves that queue/feedback order. The explicitly separate
NetSuRF adaptation probes a declared candidate subset and exchanges only
*measured* protected/unprotected bits around a frozen MSB mask at fixed global
cardinality. Unprobed bit statuses do not change. Equal feedback preserves the
baseline order. Thus results must be called **PrioriFI-derived local allocation**,
never faithful PrioriFI protection or whole-network optimization. The
repository-form Hessian is used solely to initialize probe order; the
published-form FKeras protection baseline remains a different method.

ToyCar cannot use source prediction-change feedback: the authoritative metric
is complete-file ROC-AUC. Its engineering adaptation uses positive changes in
`1 - AUC` on complete calibration files, while the continuous Hessian seed uses
reconstruction MSE. This mismatch is explicit and makes the ToyCar method
architecture/task-adapted, not source-faithful. No window-MSE result is used as
an AUC outcome.

## Pinned bounded checks

All three use the existing TF 2.14.1 `BUILTIN_REF` asset locks, exact stored
INT8 XOR and restoration, a 6.25%-coverage MSB context, 40 isolated probes in
a 60-bit presampled candidate pool, four paired Bernoulli fields at BER 0.001,
and previously exposed engineering evaluation inputs. KWS/ResNet use separate
training calibration data. ToyCar uses eight normal/anomalous file pairs from
its empirical-validation half, disjoint from four complete evaluation files.
These are not untouched publication cohorts.

| Model | Eligible bits | Probe feedback >0 | Mask symmetric difference | Hessian / probing wall time | Peak RSS | Tiny paired task-risk difference |
|---|---:|---:|---:|---:|---:|---:|
| KWS | 176,128 | 5/40 | 2 bits | 2.19 / 7.17 s | 0.87 GB | zero on 4 fields × 8 clips |
| ResNet-8 | 618,880 | 13/40 | 4 bits | 5.45 / 15.83 s | 2.29 GB | zero on 4 fields × 8 images |
| ToyCar | 2,113,536 | 12/40 | 10 bits | 8.91 / 11.45 s | 1.87 GB | zero on 4 fields × 4 complete files |

Receipts: `workspace/runs/priorifi_kws_engineering_v2`,
`workspace/runs/priorifi_resnet8_engineering_v2`, and
`workspace/runs/priorifi_toycar_engineering_v2`. Each stores lock/checkpoint
identities, sample identities and hashes, ordered probes, feedback, masks,
paired field hashes, proposed/applied/suppressed event counts, exact task
metrics, timing and a receipt checksum. The small evaluation is a restoration
and algebra check, **not** evidence that masks perform equally in general.

At BER 0.001 and 32 fields, the expected number of proposed events touching
the *changed* mask coordinates is only 0.064, 0.128 and 0.320 respectively.
Even under independence, the corresponding probabilities of observing no
changed-coordinate event are approximately 0.938, 0.880 and 0.726. A large
conventional protection campaign with these 40-probe masks would therefore
mostly compare identical faulted models. More fault fields cannot correct a
poorly resolved mask-construction protocol.

A second KWS construction-only gate sampled **736** candidate bits with known
stratum inclusion probabilities and measured **400** of them (64 training
clips, same small exposed evaluation check). It took 81.6 s, peaked at
0.83 GB RSS, and changed 26 mask coordinates (13 exchanges). The four paired
engineering fields still produced identical measured outcomes for MSB and the
derived mask. At BER 0.001, 32 fields would have only 0.832 expected proposed
events on those changed coordinates and roughly 43% probability of none.
The receipt is `workspace/runs/priorifi_kws_candidate736_probe400_v1`.
This larger gate improves mask separation but does **not** establish a useful
publication protocol or a PrioriFI advantage.

Extrapolating the measured isolated-probe cost to a source-style exhaustive
traversal gives very rough lower-order planning figures of about 9 hours for
KWS, 68 hours for ResNet-8 and 168 hours for ToyCar, **before** full-cohort
final protection evaluations, data I/O and uncertainty replication. These
are not promises of runtime: per-probe cost and model loading may change.
For the W8A8 transformers, no validated mapped continuous Hessian seed or
source-equivalent task feedback exists, and hundreds of millions of bits make
exhaustive probing prohibitive. They must be shown as NOT IMPLEMENTED for this
method, not silently replaced with gradient or significance ordering.

## Uncertainty and next decision

The current comparative database already contains matched 32-field broad
grids for KWS, ResNet-8 and ToyCar; focused published-form FKeras comparisons
also have 320 ResNet and 384 ToyCar fields per BER. Many KWS/ResNet contrasts
survive their original Holm families. ToyCar and the low-exposure W8A8 studies
remain less decisive. Trial count is not the only uncertainty: one mask seed,
one frozen model, one evaluation cohort and the ToyCar file population remain
separate replication levels. More fields at the nearly flat transformer BERs
would not locate a useful degradation transition.

For example, at BER 0.01 and 25% coverage, the saved source-paired comparison
has per-example absolute-gradient minus random ToyCar AUC-risk difference
`+0.084167`, CI `[+0.041340,+0.126995]`, original Holm `p=0.047`, while the
same comparison has strongly negative CE differences on ResNet-8 and KWS.
This is a fixed-cohort, fixed-mask contrast across different task metrics;
it supports **observed condition-specific ranker performance**, not a
universal across-model impossibility or an interaction-mechanism attribution.

Before a PrioriFI-derived production comparison, prespecify a substantially
larger candidate pool and probe budget, multiple independent candidate seeds,
the task feedback and exact fixed-budget allocator; verify nontrivial mask
separation and enough expected *changed-coordinate* events at the target BER;
then freeze masks and evaluate every method on new paired final fields. A
separate exposed-data pilot can estimate these costs and effect sizes. The
ToyCar file-exposure ledger limits fresh confirmation; new class-balanced
files may be necessary. Do not infer significance from marginal error bars.

Existing exact interaction attribution (M9.10) proves the mathematical
decomposition and shows local clean-isolated to conditional decision reversals
on bounded model-selection pools. For one KWS example the clean-isolated
contrast is +3.23349e-5 CE, while the exact contrast is -3.25639e-4 CE;
the outside-background correction is -3.66675e-4 CE. Those are local,
selection-data effects, not a whole-network causal mechanism or proof of
held-out protection superiority. The reported selected-versus-isolated
held-out pool-level intervals span zero. A defensible paper claim is that no
context-free isolated ordering has a universal guarantee, and that ranking
choice matters at measured fault conditions; the present data do **not** prove
that no single ranking works across all models or that interconnection effects
dominate independently measured damage in general.

## Reproduce only the bounded engineering checks

From this isolated worktree, with the existing pinned environment:

```bash
<assets>/.venv-mlperf/bin/python -m netsurf.priorifi_mlperf_engineering --benchmark kws --calibration-count 64 --max-probes 40 --trials 4 --output workspace/runs/<new-kws-receipt>
<assets>/.venv-mlperf/bin/python -m netsurf.priorifi_mlperf_engineering --benchmark resnet8 --calibration-count 64 --max-probes 40 --trials 4 --output workspace/runs/<new-resnet-receipt>
<assets>/.venv-mlperf/bin/python -m netsurf.priorifi_mlperf_engineering --benchmark toycar --calibration-count 16 --max-probes 40 --trials 4 --output workspace/runs/<new-toycar-receipt>
```

The destination must not exist. These commands do not run a publication study.
For a larger *construction-only* KWS diagnostic, the bounded switches are
`--sign-per-status 256 --lower-per-tier 32 --max-probes 400`; this samples
736 candidates and probes 400. The hard guard rejects more than 512 probes
in one engineering command. The 400-probe receipt above was produced before
the module was renamed from `priorifi_kws_engineering` to
`priorifi_mlperf_engineering`; the scientific path is unchanged.
