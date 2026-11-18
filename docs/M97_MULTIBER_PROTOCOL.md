# M9.7 joint KWS BER protocol

## Scientific question and status

The estimand is `J_p(P) = E_{Z~Bernoulli(p)} R(Z\P)`, where `P` is an ideal
fixed-cardinality logical protection mask and `R` is KWS cross-entropy. The
study asks how BER changes the relative value of the exact bounded
conditional selector and exact isolated-bit selector. BER is a logical
independent bit-fault probability, not a physical SEU rate. The 0.01 arm is a
new same-window comparison anchor; M9.6 outcomes at 0.01 were already known.
No M9.6 risk table is reweighted into a new BER.

The production manifest is
`workspace/runs/m97_kws_multiber_v2/campaign.json` in the `m97_ber` worktree.
It binds the twelve original M9.6 status-stratified probability-sampled K=6
pools, each candidate-list hash, the M9.6 finalized selection and final
receipts, checkpoint, asset lock, S5 mask, three BER arms, fresh seeds,
sample partitions, fixed budget, operation counts, and statistical plan. Its
SHA-256 and validation output are reported in the handoff. It is write-once.
Earlier `m97_kws_multiber_v1` is a pre-delivery frozen planning artifact;
its source evidence is retained and it must not be executed as the study.

The M9.6 source is
`<repo>/workspace/implementation-worktrees/m96/workspace/runs/m96_kws_cross_pool_k6_v1/campaign.json`.
The new code verifies its FROZEN marker, file inventory, all 24 completed
receipts and scientific bundles before copying pool identities. Candidate
sampling is never called. The twelve pool identities, including any overlap
between pools, are identical at 0.001, 0.003, and 0.01. The fixed global S5
mask protects 22,016 of 176,128 eligible kernel bits. Each candidate pool has
two originally protected and four unprotected bits; each policy protects
exactly two of six, so there are 15 eligible local masks. No result is a
whole-network optimum or a silicon mitigation guarantee.

## Inputs and exposure

The model-selection window is KWS original test indices **2152–2215**
(64 clips). The final window is **3962–4473** (512 clips), shared across all
BER arms and pools as one prespecified joint study. M9.6 final used 3450–3961.
The remaining **4474–4889** are reserved and are never read in engineering
smoke tests. The freeze scans known finalized M9.5–M9.6 run roots and the
exposure ledger; it rejects a confirmation overlap. It also loads both new
windows from the pinned TFDS order and records source split, ordered indices,
labels, encoded-input SHA-256 per clip, and partition hashes. Every selection
and final member recomputes and verifies those identities before any faulted
evaluation. The pinned adapter's 16-clip clean-reference inference has
previously occurred and is explicitly recorded; no claim of untouched clean
reference data is made. The inventory cannot prove absence of failed, manual,
concurrent, or out-of-root evaluations. A researcher must disclose any such
exposure before production execution.

## Fault law and exact calculation

For each pool and BER, 32 independent selection outside fields are drawn as
Bernoulli(p) over all eligible bits outside the six candidates and fixed S5
protection. Every field evaluates all 64 candidate states on the same 64
model-selection clips. The exact state responses are integrated with the
candidate-specific Bernoulli mass at that BER for all 15 legal masks.
The isolated oracle uses exact one-bit model-selection responses, retaining
negative as well as positive damage. The selected mask may differ by BER.

After every one of the 36 selection bundles is finalized and checksum
verified, `multiber seal` binds their selected and isolated masks into three
arm registries and a single joint SEALED marker. The final runner rejects any
arm, including a direct child `campaign run`, until this marker matches all
three registries. Each pool and BER then uses 128 new independent master
Bernoulli fields on the shared 512 final clips. The three prespecified paired
swap contrasts use exact legal paths, sum path steps within a field, and
retain signed CE effects. Each field records its seed, full event list,
schedule hashes, per-input contributions, proposed/applied/suppressed counts,
and state-restoration checks. Final schedules are independent across pools
and BERs; input clips remain shared. A failed or interrupted member leaves an
attempt record and cannot silently be restarted or assigned a new seed.
Completed finalized receipts resume only after checksum and payload checks.
Partial sealing also requires an explicit audit; no overwrite path exists.

The original S5, exact isolated, and conditional-selected masks are compared
at the same global budget:

1. conditional-selected minus original S5;
2. conditional-selected minus exact isolated;
3. exact isolated minus original S5.

Negative differences favor the first policy. A separate optional transfer
diagnostic evaluates the BER-0.01 selected mask at lower BER **on lower-BER
model-selection response tables** without reselecting it. It is descriptive,
excluded from confirmatory tests, and not a held-out transfer result. A
held-out fourth-policy transfer evaluation remains unimplemented and must be
independently frozen before running; the three-arm final evidence cannot be
silently repurposed for it.

## Prespecified analysis

The primary family consists of two two-sided, paired **pool-level** changes
in selected-minus-isolated CE: `0.001 − 0.01` and `0.003 − 0.01`. Holm adjusts
the two p-values; simultaneous 97.5% Student-t intervals are reported. The
secondary family contains the nine within-BER policy comparisons (three per
BER). Holm adjusts all nine p-values; simultaneous `1−0.05/9` Student-t
intervals are reported. All twelve pools are retained with equal weight.
Field-level intervals and favorable/adverse pool counts are separate.
The shared input window and paired pool ordering are preserved for cross-BER
differences. Fields are stochastic replicates inside pools. Individual clips,
faulted-model calls, and path steps are not independent pool samples.

Offline analysis verifies complete field and per-input records, BER and seed
identity, input identity across all arms and pools, and per-step event
accounting. It reports BER-specific masks, selection switch counts, signed
isolated damages, additive isolated-model contrast prediction errors,
selection regret among all 15 exact model-selection masks, and held-out
regret bounded to S5 and isolated competitors. It retains field-tail
quantiles and leave-one-field-out sensitivity from the exact replication
analysis; no adverse field is dropped. Small or unresolved effects at low BER
remain unresolved. There is no adaptive field increase, BER search, or
response-dependent pool selection.

## Exact planned work and resources

| Work | Per pool per BER | Per BER, 12 pools | All three BERs |
|---|---:|---:|---:|
| Selection fields | 32 | 384 | 1,152 |
| Selection candidate-state requests | 2,048 | 24,576 | 73,728 |
| Clean plus exact isolated selection requests | 7 | 84 | 252 |
| Total selection requests | 2,055 | 24,660 | 73,980 |
| Selection clip inferences | 131,520 | 1,578,240 | 4,734,720 |
| Analytical fixed-budget integrations | 480 | 5,760 | 17,280 |
| Final paired master fields | 128 | 1,536 | 4,608 |
| Final paired-state requests, upper bound | 1,536 | 18,432 | 55,296 |
| Final clip inferences, upper bound | 786,432 | 9,437,184 | 28,311,552 |

The total conservative request upper bound is **129,276** states and
**33,046,272** clip inferences, plus 16 bounded clean-reference clips per
adapter load. Exact path and cache reuse lower actual final calls. These
counts are a plan, not a formal power calculation. Under the natural law,
expected proposed bit events per persistent field are `176128p`; expected
applied events after the 22,016-bit mask are `154112p`. At BER 0.001,
0.003, and 0.01 these are respectively **176.128/154.112**,
**528.384/462.336**, and **1761.28/1541.12**. Selection outside-field
expected counts are `154108p`, and pre-policy final master-field expected
counts are `154114p`. Query-specific observed proposed, applied and
suppressed events are saved in finalized member bundles and summarized
offline. Forced swap-bit queries are not natural Bernoulli trials.

The completed M9.6 production campaign took 5,840 s for twelve selections
and 7,462 s for twelve finals on its measured host. At the same throughput,
three sequential arms project about **4.87 hours selection** and **6.22
hours final**, about **11.1 hours total**, excluding new overhead and
offline analysis. Treat 12–15 CPU-hours as a planning allowance, not a
guarantee. Run one member at a time; allow 1–2 GiB memory and roughly
6–15 GiB free disk as a cautious three-arm planning allowance. The M9.6
12-pool results and exact size of new event records should be checked before
allocating storage. If cost is constraining, execute **selection only**
across the entire frozen grid first, then decide whether to fund final
evaluation without inspecting final effects or editing the frozen design.

## Foreground researcher commands

Use the pinned TF 2.14.1 reference environment for selection and final. Do
not use modern TF 2.21 to claim equivalent scientific results. Do not run
these production commands automatically from an agent or background job.

```bash
cd <repo>/workspace/implementation-worktrees/m97_ber
M97=workspace/runs/m97_kws_multiber_v2/campaign.json
PY=<assets>/.venv-mlperf/bin/python

$PY -m netsurf multiber validate "$M97"
$PY -m netsurf multiber run "$M97" --phase selection
$PY -m netsurf multiber seal "$M97"
$PY -m netsurf multiber run "$M97" --phase final
<netsurf-env>/bin/python -m netsurf multiber aggregate "$M97" \
  --output workspace/runs/m97_kws_multiber_analysis_v1
```

For operational batches, selection or final may add `--ber 0.001` and/or
`--pools pool_000 pool_001 pool_002`. `seal` requires all 36 selections.
Rich progress is enabled where available; `--no-progress` keeps plain
foreground output. A finalized member can be resumed through the same
command without rerunning inference. Never rename an interrupted member to
obtain a fresh sample identity.

## Engineering demonstration and export

`workspace/runs/m97_engineering_v1` used the original M9.6 pool 000, BER
0.003, two selection fields, two final fields, and two previously exposed
clips per phase (selection 2088–2089; final 3450–3451). Its finalized
analysis is `workspace/runs/m97_engineering_analysis_v2` (the initial
offline derivation v1 is retained unchanged). It validates
selection, seal, final, checksums, pairing, mask cardinality and state
restoration. It is not evidence of superiority or confirmatory generalization.
Its two-field selected-minus-S5 CE mean was `-2.9703e-6`; the tiny sample
does not support a directional claim. Mean proposed/applied events per
recorded paired-step request were both 459.5, with zero suppression because
the exact forced-step schedules exclude the fixed protected bits. The
natural-law expectations above are different quantities.

Focused M9.7/M9.6 regressions: **28 passed, 2 skipped**. The full
Python-3.11 `netsurf` test command produced **309 passed, 9 skipped, 6
failed** in 75 s. The six are the same pre-existing external-state failures
listed in M9.6: four missing worktree-local MNIST checkpoint paths, one
stale IC asset-lock digest expectation, and one IC checkpoint path mismatch.
The private `proj` shell activation could not start because Lmod is absent;
the exact registered `netsurf` Python interpreter was invoked directly.
No asset lock, historical test, or checkpoint was modified to hide these.

After production completion, create a ZIP of all scientific evidence in the
worktree, including campaign, every arm's plans/receipts/bundles and the
offline analysis. The engineering archive can be created now. Avoid adding
checkpoints or datasets; pinned paths and hashes remain required to rerun.
The existing `workspace/exports/m97_handoff.zip` contains source changes,
this protocol, the v2 production plan, and the finalized tiny engineering
evidence. Verify its SHA-256 after transfer.

```bash
cd <repo>/workspace/implementation-worktrees/m97_ber
mkdir -p workspace/exports
# Existing handoff bundle can be checked before transfer:
sha256sum workspace/exports/m97_handoff.zip

zip -r workspace/exports/m97_engineering_evidence.zip \
  workspace/runs/m97_engineering_v1 workspace/runs/m97_engineering_analysis_v2
sha256sum workspace/exports/m97_engineering_evidence.zip

# Only after all production phases and offline aggregation have finalized:
zip -r workspace/exports/m97_complete_scientific_evidence.zip \
  workspace/runs/m97_kws_multiber_v2 workspace/runs/m97_kws_multiber_analysis_v1
sha256sum workspace/exports/m97_complete_scientific_evidence.zip
```

On the Mac, with the researcher's configured SSH alias:

```bash
scp kona:<repo>/workspace/implementation-worktrees/m97_ber/workspace/exports/m97_engineering_evidence.zip .
shasum -a 256 m97_engineering_evidence.zip
unzip -l m97_engineering_evidence.zip

# Source, final frozen plan, and engineering demonstration are available now:
scp kona:<repo>/workspace/implementation-worktrees/m97_ber/workspace/exports/m97_handoff.zip .
shasum -a 256 m97_handoff.zip
unzip -l m97_handoff.zip

# After production completion:
scp kona:<repo>/workspace/implementation-worktrees/m97_ber/workspace/exports/m97_complete_scientific_evidence.zip .
shasum -a 256 m97_complete_scientific_evidence.zip
unzip -l m97_complete_scientific_evidence.zip
```

## Limits

The exposure scan covers known finalized evidence and a frozen ledger, not
unrecorded or concurrently run jobs. Shared contiguous TFDS clips make
input-resampling sensitivity approximate. Twelve candidate pools do not
identify a whole-network optimum or guarantee population-wide improvement.
The family has no formal power calculation; low-BER effects may be below the
resolution of 128 fields. Exact paired-swap expectations use Monte Carlo
outside fields, so finite-field error remains. No physical ECC, TMR,
silicon, or radiation claim follows from ideal logical event suppression.
