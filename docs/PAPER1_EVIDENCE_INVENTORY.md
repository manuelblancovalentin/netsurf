# Paper 1 evidence inventory and reconstructable engineering curves

Date: 2 November 2026. This is **derived analysis of finalized evidence**, not new
scientific execution. All rows/BERs/methods in the three selected static grids are
retained. Selection of these grids is based on the largest common classification
cohorts and the repaired replicated ToyCar study, not on favorable outcomes.

## 1. Fresh verification, provenance and reproduction

Implementation worktree:
`/projects/netsurf/manuelbv/workspace/implementation-worktrees/paper1-engineering`.
Derived output:
`workspace/runs/paper1_reconstruction_audit_20261102_v4/` within that worktree.

- `verification.json`: 125 finalized records, 592 checked files, full source paths,
  inventory hashes, canonical payload hashes and the verification scope.
- `static_inventory.json`: model, checkpoint, dataset, sample identities,
  sample hashes, runtime, population-related controls and source sizes.
- `static_curves.csv`: 280 condition rows, including controls, recomputed directly
  from all 7,616 trial rows.
- `paired_comparisons_checked.csv`: 1,680 saved paired comparisons, independently
  checked for paired mean, SD, t interval and number of matching fields.
- `previews/`: six readable budget/BER plot previews (PNG and PDF), with ordinary
  method names and authoritative task metrics. These are draft-design evidence,
  not a replacement manuscript or evidence of general superiority.

The earlier draft's `evidence/sources.json` supplies **locators only**. Every listed
consumed file is rehashed against its original inventory, each canonical payload
is recomputed, and source inventory identity is checked. The ToyCar exposure
ledger uses a finalized payload receipt instead of a separate manifest; that
format is explicitly validated. Raw static trials, configs and the three large
mask artifacts were additionally hashed. No giant prediction CSV was loaded.

Pairing is `(logical BER, replication, trial, schedule_sha256)`, never trial alone.
All conditions at a BER must have exactly matching identities and proposed event
counts. Proposed = applied + suppressed is checked for every row. Full protection
has zero applied events and constant clean outputs; unprotected rows suppress none.
The clean/full-control values agree with each replication's saved clean evaluation.
Recorded calibration/validation/evaluation identities are disjoint by source split
and index, and every reported cardinality matches coverage times eligible population.
Published protected cardinalities come from verified aggregate/payload records;
this pass hashes, rather than reparses, every identity in the large mask files.
Adapter/context validation in the original runs provides the eligibility checks.

Run the offline audit again into a **new** destination:

```bash
cd /projects/netsurf/manuelbv/workspace/implementation-worktrees/paper1-engineering
/mnt/raid0/ml/conda_envs/manuelbv/envs/netsurf/bin/python workflows/paper1_evidence_audit.py \
  --sources /projects/netsurf/manuelbv/publications/tecs_internal_review_2026-11-02/evidence/sources.json \
  --output workspace/runs/paper1_audit_independent_check
/mnt/raid0/ml/conda_envs/manuelbv/envs/netsurf/bin/python workflows/paper1_curve_preview.py \
  workspace/runs/paper1_audit_independent_check
```

These commands read evidence and create new derived tables/figures only; they do
not import TensorFlow or run a model. Earlier development attempts and previews
are not the delivered evidence package; the complete deliverable is explicitly `_v4`.
The derived tables are not a re-finalization or alteration of the scientific runs.

## 2. Main static evidence: enough for the first engineering draft

All models use frozen INT8 kernel codes and authoritative TF Lite **2.14.1
BUILTIN_REF**, one reference thread, no delegates. Biases, scales, zero points and
activations are excluded from the eligible storage population. Allocated zeros
remain eligible. The score companion is not the authoritative faulted forward pass.

| Property | ResNet-8 / CIFAR-10 | DS-CNN / Speech Commands | ToyCar autoencoder |
|---|---:|---:|---:|
| Eligible kernel words / bits | 77,360 / 618,880 | 22,016 / 176,128 | 264,192 / 2,113,536 |
| Protected bits at 12.5% / 25% | 77,360 / 154,720 | 22,016 / 44,032 | 264,192 / 528,384 |
| Evaluation population | 1,000 fixed test images | 1,000 official-subset clips | 64 complete files, 12,544 windows |
| Ranking population | 64 training images | 64 training clips | 16 training reconstruction windows |
| Clean task metric | accuracy 0.880000 | accuracy 0.901000 | file ROC-AUC 0.869140625 |
| Clean auxiliary loss | CE 0.588299415 | CE 0.379391470 | reconstruction MSE 10.928793892 |
| Fault fields per BER | 4 replications × 8 = 32 | 4 × 8 = 32 | 2 × 4 = 8 |
| Independently realized random masks | 4 | 4 | 2 |
| Other ranking masks per budget | 1 fixed mask each | 1 fixed mask each | 1 fixed mask each |
| Interior coverages | 12.5%, 25% | 12.5%, 25% | 12.5%, 25% |
| Additional controls | 0%, 100% | 0%, 100% | 0%, 100% |
| BER grid | 1e−5, 1e−4, 3e−4, .001, .003, .01, .03, .1 | same eight values | 1e−5, 1e−4, .001, .01 |
| Faulted mask/field evaluation rows | 3,584 | 3,584 | 448 |
| Distinct proposed fields over grid | 256 | 256 | 32 |

Each grid contains random, corrected MSB-first, numerical jump, global first-order,
per-example absolute and per-example positive ranking. Budgets are logical bit
counts, not percentages of chip area. At BER .001 expected unprotected flips are
618.88, 176.128 and 2,113.536 respectively; at 25% coverage the expectation is
75% of each count. Equal BER is not equal absolute exposure. Actual proposed and
applied means are retained for every condition in the derived CSV.

Classification cohorts are disjoint from gradient calibration by source split;
ToyCar ranking uses training windows and evaluation uses complete official test
files. Exact identities/hashes are in `static_inventory.json`. These cohorts have
subsequently been repeatedly examined. They are not fresh confirmation inputs.
The ResNet 1,000-image grid reuses schedule seeds from the smaller preceding study;
the two studies cannot be pooled as independent fault replication.

### Exact source locators and scientific payloads

All paths below are relative to `/projects/netsurf/manuelbv/workspace/runs/`:

| Model | Finalized directory | Scientific payload SHA-256 |
|---|---|---|
| ResNet-8 | `m92_resnet8_static_test1000_v1/20261029T084524.633748Z-f0bcc564` | `3ed95baa04ede8047f5b5eb04f9024f2e5f856c39abcc84b2ca1fb894b22db32` |
| KWS | `m93_kws_static_ber_sweep1000_v1/20261029T183410.488907Z-32c31070` | `b7b725fd1c196f2d5896b67aba34dfe8aa864b8308875df098d48be6135975e0` |
| ToyCar | `m93_toycar_static_ber_pilot64_parallel_repair_v1/20261029T234704.724222Z-3913968c` | `5d9c1eaa69e24b5f2139b7bf69ae0d775cc8727d458d0b66c5cfe897ab44b554` |

Checkpoint SHA-256:

- ResNet-8: `3c002613d1b2475eb51dd78dfb85a546c8ae658dee71cf6ade43b022fe205415`.
- KWS: `aeea436800704fce17b17292e4412630ad856e9d777c044c64ef748a880bd0ae`.
- ToyCar: `87cf24194ef93d1d9b11a591d805526b98008e351655d29883c825c9c106ba24`.

The historical 200-image ResNet pilot (`461c5d6712c3…`), 200-image BER grid
(`458c35f5603f…`), KWS 200-clip stress smoke (`dce5f4ade8cf…`), ToyCar 64-file
diagnostic (`2f580f5499d8…`) and single-replication study (`4e6f1a050005…`) were
located in the canonical archive. Smaller M8/M9 and two-file ToyCar tests are
engineering controls. They are not silently pooled with the main grids. Their
existence/metadata were inspected; fresh complete consumed-file verification in
this delivery concerns the explicitly listed 125 sources, not every older smoke.

## 3. Effects must include unfavorable and unresolved outcomes

For illustration of the **same** stress condition across all three tasks, the
table below reports per-example absolute minus corrected MSB at BER .01. Accuracy
effects are percentage points; ToyCar effects are **AUC units**. This is not a
pooled cross-task effect. All other BERs/methods remain in the accompanying CSVs.

| Task | Budget | Paired effect | Marginal 95% field interval | Saved within-BER/metric Holm p |
|---|---:|---:|---|---:|
| ResNet-8 accuracy | 12.5% | +33.0063 pp | [27.8218, 38.1907] pp | .00420 |
| ResNet-8 accuracy | 25% | +6.6813 pp | [4.4898, 8.8727] pp | .00420 |
| KWS accuracy | 12.5% | +3.5219 pp | [1.0978, 5.9460] pp | .05839 |
| KWS accuracy | 25% | +0.2063 pp | [−0.6573, 1.0698] pp | 1.00000 |
| ToyCar file AUC | 12.5% | −0.005249 | [−0.129309, +0.118811] | 1.00000 |
| ToyCar file AUC | 25% | +0.241455 | [+0.059277, +0.423633] | .84375 |

Thus a positive marginal interval is not automatically familywise significance.
The ToyCar result is particularly underpowered (eight fields), and its budget/BER
curves are nonmonotonic. Do not flatten, clip away or explain these away as noise
without analysis. Negative protection benefit is allowed by the objective.

Intervals describe fault-field variation on the fixed checkpoint, cohort and
realized masks. Random masks vary at the replication level; 32 fields do not mean
32 independently generated random masks. The preview's pooled t intervals are
descriptive, not an uncertainty statement about the population of random rankings.
Repeated full-protection controls are deterministic checks, not new samples of
clean-model performance. No marginal interval comparison replaces a paired test.

## 4. Isolated-susceptibility and computation evidence

The three finalized balanced-stratum studies have 320 candidates each (four per
kernel×bit stratum), shared across estimators. Their payloads are:

- ResNet-8 `c38df68e491f5bdc96b99ac11343fa2688d3c3bc17c12ec1dd3a31d04c41d4c5`.
- KWS `aba8b17abd5457dc22527be76728dd1b40524444fda1f0cc09a273c3d5876fde`.
- ToyCar `08e13ca8e4edc532b50fece43a4c5cd9134e631455ae5f27a466d8e793cf76ec`.

Positive-damage Spearman associations, rounded from verified saved summaries:

| Method | ResNet CE | KWS CE | ToyCar **MSE**, not AUC |
|---|---:|---:|---:|
| Corrected MSB | .256 | .204 | .502 |
| Numerical jump | .070 | .175 | .439 |
| Global first-order | .364 | .258 | .498 |
| Per-example absolute | .376 | .288 | .546 |
| Per-example positive | .364 | .246 | .637 |
| Random reference | −.054 | .012 | .059 |

Use the saved within-stratum bootstrap intervals and tie-aware top-k values, not
these point associations alone. These estimate the sampled design, not uniform
whole-network top-k performance. A single random reference is not the expectation
over all random orderings. Signed loss changes, positive damage and absolute
changes must remain separate objectives. Companion class agreement is imperfect:
the ResNet audit has 197/200 agreeing classes and a maximum probability difference
of .308594. Correct variable mapping does not make STE gradients derivatives of
integer inference.

Measured global-gradient construction times were 1.213, .419 and 3.881 seconds;
shared per-example absolute/positive computation was 17.170, 8.314 and 205.882
seconds, respectively. Shared work must not be counted twice. These are recorded
construction costs for those study configurations, not benchmark-independent
complexity constants or whole experiment time. A recorded zero baseline timer
does not prove a zero-cost implementation.

FKeras's finalized KWS smoke has payload
`2db342a2f4cb38cf553124cb10eba115d566255d84afec6b9e82d97a14e29b5e`.
Two calibration clips, four test clips and two fields validate mapping/solver/
mask invariants only; every protection condition had accuracy 1.0. The partial
6.25%/18.75% paper/repository masks differ (symmetric difference 1,802 bits), but
this is not performance evidence. Whole-tier 12.5%/25% masks equal corrected MSB.

## 5. Conditional results: retain the limitation, not the old draft's emphasis

The verified all-pool attribution bundle at
`workspace/implementation-worktrees/m910-interaction-attribution/workspace/runs/m910_production_attribution_committed`
has payload `cb9099abed7a74c06dcae1c96a8f1122c3a8efdb6527e1a28475560601399ace`.
All its 112 source selection/final bundles were freshly checked. The **equal-pool
held-out selected-minus-exact-isolated intervals include zero in all six arms**:

| Arm | Mean risk difference | 95% pool interval |
|---|---:|---|
| KWS first independent pools, BER .01 | −2.02436e−4 CE | [−4.07816e−4, +2.94429e−6] |
| KWS common pools, BER .001 | −3.39827e−7 CE | [−7.67165e−6, +6.99200e−6] |
| KWS common pools, BER .003 | −1.45116e−5 CE | [−3.99828e−5, +1.09596e−5] |
| KWS common pools, BER .01 | −1.22785e−4 CE | [−2.88804e−4, +4.32347e−5] |
| ResNet-8, BER .01 | +1.10811e−4 CE | [−1.68151e−4, +3.89772e−4] |
| ToyCar, BER .001 | +7.10720e−6 AUC risk | [−9.95590e−6, +2.41703e−5] |

The earlier frozen-mask KWS replication is exploratory because the candidate pool
was historically selected. Its favorable result is not erased, but cannot replace
these independently sampled pool results. Common KWS pools across BERs are not
independent pools. Rank-reversal pairs within a pool are not independent trials.
Signed decomposition on selection response tables does not establish causal
attribution of held-out improvements. All-candidate interaction orders are retained
in the bounded exact response table; that does not confer whole-network optimality.

M11's verified offline reanalysis and bounded engineering checks provide identities,
approximation error/regret checks and worst-case bounds—not executed production
scaling. Some rank changes do not change the selected mask; finite-selection-field
regret can be zero. Include these negative findings briefly, proofs/limits in an
appendix. Do not equate a mathematically valid worst-case bound with a useful
numerical certificate at whole-network dimension.

## 6. Availability, gaps and resource basis

**Available now:** all three models' main budget/BER curves, paired comparisons,
controls, raw measurements, sampled estimator validity/cost, bounded conditional
diagnostics, actual failure/null results. **Missing:** 6.25%/18.75% static cells,
adequately powered FKeras protection tests, ToyCar BER .003 and more fields,
broader independently trained checkpoints, physical hardening costs.

Observed coordinator elapsed time / output size: ResNet 12,018.68 s / 2.505 GB;
KWS 2,701.47 s / .718 GB; ToyCar 1,288.18 s / 4.044 GB. Requested/effective workers
were 2/2, 8/4 and 2/2. ToyCar workers peaked near 15.1 GB aggregate RSS; score/mask
serialization is a material cost. Use monotonic coordinator timing, not the much
shorter coordinator-local start/completion timestamps, which do not cover the full
worker phase. This reporting limitation does not alter scientific payloads.

The long-term registry retains 11 families/18 precision targets. Only ResNet-8,
DS-CNN and ToyCar have the main static/conditional evidence used here. Streaming
1D DS-CNN has a short engineering fixture, not a study-ready task/partition;
VWW image provenance is unresolved. ResNet-18, MobileNetV3, ViT, DistilBERT, Pythia
and Qwen targets remain asset/precision/backend gates, not completed experiments.
They are not prerequisites to this internal draft.
