# M9.5.6 publication preflight

This is an infrastructure and engineering gate, not an effectiveness study.
Implementation branch: `m956-publication-preflight`, isolated from
`m955-paired-swap` at `afe5424c75923e14f2cb38fc85932c3c2fbf660d`.
No historical bundle, asset lock, or earlier TOML is modified.

## Final-evaluation exposure audit

The audit verified every inventoried artifact checksum in all seven finalized
M9.5.3–M9.5.5 smoke executions under their implementation workspaces. The
canonical run root and M95 worktree contain no additional finalized M953–M955
runs. `netsurf/data/kws_prior_exposure_v1.json` preserves each exact bundle path,
payload hash, checksums-file hash, partition hash, original sample ID, label and
encoded-input hash. Duplicate executions do not make samples fresh again.

| Milestone | Final original TFDS test indices | Executions | Final partition SHA-256 |
|---|---|---:|---|
| M953 | 3048, 3049, 3050, 3051, 3052, 3053 | 3 | `dc611f68262dcdd74bf1487a117cbdedfddd1832db596370b6aee349bf326059` |
| M954 | 3054, 3055, 3056, 3057, 3058, 3059 | 2 | `c45a18a2f7efcb91307bc11c924b93b195d1fa4d65facfaa711634a25d708052` |
| M955 | 3060, 3061, 3062, 3063, 3064, 3065 | 2 | `5b40f584fef040f82ae9304086d3955a742683ee04b0130dc9187fd20fede1e7` |

All seven executions are individually listed in the machine-readable ledger. The first M953
execution had a different payload, but used the same six final inputs.

`m955_kws_s5_context_k8.toml` starts at relative final offset 12, so its first
six examples overlap M955 validation. That historical TOML remains unchanged.
The new templates reserve relative offset **18**, count **128**, hence original
indices **3066–3193**, disjoint from every fault-evaluated smoke input in this
ledger. Calibration is 1000–1031 (inactive in these modes); model selection is
2024–2087. These are fixed windows in the pinned round-robin-block-16 source
order, not evidence that their class distributions are identical.

"Unexposed" means no prior fault-outcome evaluation in the audited records.
All 4,890 clips previously underwent aggregate clean-reference evaluation;
window labels were also audited in M954. It does not mean inference-naive data.
The original windows' class-count ranges were 74–104, 66–105 and 135–174;
see M954 for their label-bound hashes. No new final labels or inputs were read
to tune the publication window in this milestone.

Schema 6.4 rejects confirmation plans intersecting the ledger, before adapter
initialization. It also rejects overlaps with finalized 6.4 final manifests
under the configured output root. An explicit `purpose="engineering"` permits
known exposure but records it and never claims confirmation. This is not a
global registry, a cross-process reservation, or a detector of unpublished
failed/manual inference. Re-audit new runs elsewhere before publication; do
not execute concurrent confirmation runs on the same reserved window.

## Candidate provenance: exploratory, frozen, not random

The eight candidates are exactly the first eight entries of the ten-bit M95
smoke list introduced in `fdad54f`; M953 retained this prefix. Their encoded
positions span LSB through sign bit. No probability sampling rule, original
word-selection seed, or inclusion probabilities are recorded. Thus the
verifiable rule is **retain the first eight historical explicit identities**,
not uniform sampling or an optimized whole-network population.

| Index | Exact tensor/word/bit identity | Original S5 protected? |
|---:|---|---|
| 0 | `tflite_tensor_0005/kernel[295]#b0` | No |
| 1 | `tflite_tensor_0008/kernel[105]#b1` | No |
| 2 | `tflite_tensor_0011/kernel[511]#b2` | No |
| 3 | `tflite_tensor_0014/kernel[485]#b3` | No |
| 4 | `tflite_tensor_0016/kernel[35]#b4` | No |
| 5 | `tflite_tensor_0017/kernel[2525]#b5` | No |
| 6 | `tflite_tensor_0018/kernel[497]#b6` | Yes |
| 7 | `tflite_tensor_0019/kernel[442]#b7` | Yes |

Membership was recomputed from the verified source S5 mask (22,016/176,128
bits). Original local mask is `(6,7)`; outside-context cardinality is 22,014.
Every eligible budget-two exchange retains the global 22,016-bit budget.
These tensor identifiers resolve directly to the pinned model's INT8 kernel
buffers; biases, scales and graph metadata are not in the eligible population.

Earlier outcomes were known when this pool was reused: the M953 task explicitly
highlighted bit `0018[497]#b6` because its M952 isolated and conditional damage
had different signs. The original identities predate that result, but there
is no evidence establishing outcome-independent selection of the later prefix
or research hypothesis. Nor can the missing word-selection provenance be
reconstructed from the TOMLs. We therefore classify the retained pool and
mechanism hypothesis as **exploratory**. Independent final inputs/backgrounds
can assess these frozen masks; they cannot remove pool-selection bias or
establish generalization to arbitrary network bits.

### Separate later-confirmation pool protocol (not executed)

Before inspecting any new bit-damage outcomes: freeze the checkpoint, kernel
universe hash, source S5 mask hash, BER, task metric, budget and pool-size rule.
Draw two protected and six unprotected identities uniformly without replacement
from their respective complete populations using a published new seed. Record
the RNG/version, sorted populations, draws and inclusion probabilities
`2/22016` and `6/154112`. Do not reject a draw for weak effects, tensor imbalance,
or agreement with S5. This is a protection-status-stratified design, not a
uniform whole-network sample. Freeze an independent selection partition,
new final sample ledger and final schedules before measurements. Multiple
prespecified pools would be needed for pool-level generalization; do not treat
outside fields within one pool as independent pool replications. This protocol
is documented only, not implemented as another sampler or optimizer.

## Schema 6.4: two uses of the existing engine

`schema_version="6.4.0"`,
`scientific_protocol="frozen-selection-protection/v5"`.

New `[workflow]` fields:

- `mode`: `selection_only` or `final_only`.
- `purpose`: `engineering` or `confirmation` (exposure gate, not a claim of
  population-representative candidates).
- `candidate_provenance`: `exploratory_frozen_m953_pool`.
- Final-only requires `selection_bundle`, `selection_payload_sha256`, and
  `selection_artifact_sha256` (file SHA-256 of `artifacts/frozen_selection.json`).

The remaining scientific keys keep schema-6.3 meanings: `[samples]`, explicit
`[[candidates]]`, `[backgrounds].probabilities`, `[restricted_oracle].budget`,
`calibration_background_count`, `calibration_replicate_seeds`, the verified
`[protection_context]`, `[paired_swap_evaluation]` and `[execution].confidence`.
This preflight supports exactly one prespecified BER. Legacy descriptive
calibration/evaluation background fields and full-final-enumeration fields
remain syntactically present for compatibility, but are **not executed** in
either new mode. Their inactive sample/seed settings are not measurements.
Resolved operation counts and execution provenance make this explicit.

Selection-only disables `[paired_swap_evaluation].enabled`, but its count,
seed, comparisons and final sample settings prespecify the later final plan.
Only model-selection inputs are loaded. The mandatory clean adapter gate still
uses official reference clips 0–15; it does not access the reserved final window.
The full-window label audit is disabled in both separated modes.

The original exact selector is now callable without held-out responses;
joint schema 6.0–6.3 behavior is retained. Each of the `2^K` responses is saved,
and all mask risks use the same exact Bernoulli integrator. Individual-field
winners/margins, cumulative winners/margins/switches, per-mask field risks,
and replicate-level winners/stability are saved. This is not a new optimizer.

Final-only checks the finalized marker, entire checksum inventory, canonical
payload hash, frozen-artifact file hash and its payload binding. It requires
matching model/lock, exact pool order, eligibility hash, source context,
budget, BER, risk, selection plan and final plan. No model-selection partition
is requested and no selection inference, response tables, or final post-hoc
optimum are computed. Sequential TFRecord access still reads earlier raw
records to locate a requested window; it does not evaluate those earlier clips.
Final masks are loaded exactly
once and their indices/cardinalities and source-S5 membership are validated.
Missing or stale artifacts are errors, never a request to reselect.

## Prespecified pilot and final plan

`config/examples/m956_kws_s5_selection_k8.toml`:

- K=8, local budget 2, global S5 budget 22,016, BER 0.01.
- 64 model-selection samples (original 2024–2087).
- Eight fields for each of seeds `[956711,956712,956713,956714]`: **32** fields.
- Exact field seed is `base_seed + local_field * 1000003`; derived seed
  collisions and overlap with final field seeds are rejected.
- All 28 fixed-budget masks integrated for every response table.
- **8,201** state requests: `9 + 32*256`; **524,864** clip inferences at 64
  clips/request, plus setup/reference checks. 896 mask-risk integrations and
  12,096 pairwise table differences require no extra model calls.

`config/examples/m956_kws_s5_final_k8.toml`:

- Same pool, context, BER and masks supplied by the frozen pilot artifact.
- Final original test indices 3066–3193; 128 clips.
- **64** independent master fields, base seed **956901**.
- Primary metric: expected categorical-cross-entropy difference, target minus
  baseline; negative favors target.
- Ordered family: selected–original S5 (primary), selected–exact isolated,
  exact isolated–original S5 (secondary).
- Two-sided 95% Student-t marginal intervals on whole-path field contrasts;
  additional Bonferroni intervals cover the three-contrast family. No
  automatic significance or estimator-superiority declaration. Uncertainty
  is conditional on the fixed model, inputs, pool and selected masks.
- At most **768** state requests (`64*3*2*2`), hence **98,304** clip inferences
  plus setup. Actual paths/caches may reduce this. Identical masks have exactly
  zero contrast and require no faulted inference for that contrast.

These field counts are pilot/planning choices, not a power claim. Do not alter
the pool, mask, BER, final samples, seeds or comparison family after reading
final outcomes. Changing a frozen plan requires a new selection artifact and
a re-audited independent final plan, not patching an old bundle.

## Researcher commands

Use this isolated worktree; nothing has been merged into `v2`:

```bash
cd /projects/netsurf/manuelbv/workspace/implementation-worktrees/m956
export NETSURF_WORKSPACE="$PWD/workspace"
/projects/kona-assets/manuelbv/.venv-mlperf/bin/python -m netsurf conditional validate \
  config/examples/m956_kws_s5_selection_k8.toml --resolved
```

After reviewing the 8,201-request plan, run the pilot in tmux yourself:

```bash
/projects/kona-assets/manuelbv/.venv-mlperf/bin/python -m netsurf conditional run \
  config/examples/m956_kws_s5_selection_k8.toml
```

Inspect its report and selection stability without running final evaluation.
To bind the final template, copy the reported bundle's absolute path into
`selection_bundle`, its CLI/manifest payload digest into
`selection_payload_sha256`, and the following file hash into
`selection_artifact_sha256`:

```bash
sha256sum /ABSOLUTE/SELECTION/RUN/artifacts/frozen_selection.json
```

The final template deliberately contains `REQUIRED_*` placeholders and **fails
validation until bound**. It never guesses a run path. Then:

```bash
/projects/kona-assets/manuelbv/.venv-mlperf/bin/python -m netsurf conditional validate \
  config/examples/m956_kws_s5_final_k8.toml --resolved
/projects/kona-assets/manuelbv/.venv-mlperf/bin/python -m netsurf conditional run \
  config/examples/m956_kws_s5_final_k8.toml
```

Required baseline: the hash-pinned M93 S5 source bundle
`/projects/netsurf/manuelbv/workspace/runs/m93_kws_static_ber_sweep1000_v1/20261029T183410.488907Z-32c31070`,
the pinned KWS lock, model and Speech Commands data remain at the locations
in the TOMLs. Modern TF2.21 is not a replacement for the TF2.14.1 BUILTIN_REF
interpreter. No packages or toolchains were changed.

## Bounded validation evidence

Only engineering runs were executed. The K=8 smoke uses two model-selection
clips (2024–2025), two outside fields under seeds 956111/956112 and two
previously exposed final clips (3060–3061), with three final master fields
under 956211. It uses the same eight candidate identities and budget two.

Initial finalized engineering evidence under this worktree's `workspace/runs/`:

- Selection: `m956_engineering_selection_k8/20261030T155653.364948Z-3973bfca`,
  payload `bc0b1b8af3261ccb202b9f35dd12161c624697deee158faa17b66083392d19b8`.
- Final: `m956_engineering_final_k8/20261030T155847.663786Z-27c73633`,
  payload `5b4efe74ccf9b81f9e685b1203d88168d9a4ba7acb2054faedf12d857b0d3556`.

Selection took 57.59 seconds including initialization/preprocessing, requested
521 state evaluations and loaded no reserved final inputs. Selected and
original masks both were `(6,7)`; isolated was `(0,1)`. Separate final loading
took 51.03 seconds including initialization, used 24 path requests and 12
unique authoritative inferences shared across comparisons. Selected–original
was identically zero. No effectiveness inference follows from two clips and
three fields. Startup dominates these tiny timings; proportional scaling of
total wall time is not an accurate publication runtime estimate. Request and
clip-inference counts above are the reliable cost bounds; allow hours for the
pilot and monitor actual progress before increasing counts.

The optional pinned-runtime test independently verifies a two-candidate
selection/final sequence and asserts that forbidden sample roles cannot be
accessed. Synthetic tests cover additive, beneficial and higher-order risks,
equivalence to the previous selector, deterministic payloads, field-seed
collisions, artifact corruption, changed plans, sample-exposure rejection and
frozen-mask invariance. Test counts and final implementation commit are recorded
in the delivery summary; external environment failures are not concealed.
