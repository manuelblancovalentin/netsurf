# M9.5.7 independent frozen-mask replication

Implementation branch: `m957-frozen-mask-replication`, worktree
`/projects/netsurf/manuelbv/workspace/implementation-worktrees/m957`, based on
M956 `9a96d5e2267fac5fa113ec9d4043ac9311f6b707`. No merge or push.
This prepares an independently specified replication. **The 256-field
publication replication has not been executed.**

## Verified source and prior result

The original finalized selection is:

`/projects/netsurf/manuelbv/workspace/implementation-worktrees/m956/workspace/runs/m956_kws_s5_selection_k8/20261030T165154.608989Z-b465f389`

- Scientific payload: `6acf2a45452432e64186f52c3ffa6dfd2f298765106c1282ee220ef5823f0ff7`.
- Frozen-selection file SHA-256: `675ac2b3a46a1555fb0f4da86a0808c96ca2b4ad961e66b68a4284f97286b046`.
- Selected `(2,6)`, exact isolated `(2,7)`, original S5 `(6,7)`.
- Global allocation 22,016/176,128 INT8 kernel bits, fixed outside mask 22,014.
- Model SHA-256: `aeea436800704fce17b17292e4412630ad856e9d777c044c64ef748a880bd0ae`.
- KWS lock file SHA-256: `09fc5867d721363a0156e2f9f6fdedefe2aa0ae7384cc72396bc594e283bfac7`.

The original final bundle is
`.../m956/workspace/runs/m956_kws_s5_final_k8/20261030T170754.064906Z-5ee152da`,
payload `d1726e5be00efcb14382dfc2d26fd5921cee6a8e43fc9a590c7ca178ac8cbf93`.
Its complete conditional-bundle inventory was reverified. Saved whole-field
summaries reproduce the requested prior results:

| Contrast | Fields | Mean CE difference | Marginal 95% CI |
|---|---:|---:|---|
| Selected − original S5 | 64 | −0.0001559260694 | [−0.0003505537213, +0.00003870158249] |
| Selected − isolated | 64 | −0.0001528340563 | [−0.0003110696215, +0.000005401508868] |
| Isolated − original S5 | 64 | −0.000003092013102 | [−0.00008160157492, +0.00007541754872] |

These are prior evidence, not new replication outcomes. No mask, seed, count
or hypothesis was changed in response to new final outcomes.

## Exact frozen pool

| Index | Bit identity | Original S5 protected |
|---:|---|---|
| 0 | `tflite_tensor_0005/kernel[295]#b0` | No |
| 1 | `tflite_tensor_0008/kernel[105]#b1` | No |
| 2 | `tflite_tensor_0011/kernel[511]#b2` | No |
| 3 | `tflite_tensor_0014/kernel[485]#b3` | No |
| 4 | `tflite_tensor_0016/kernel[35]#b4` | No |
| 5 | `tflite_tensor_0017/kernel[2525]#b5` | No |
| 6 | `tflite_tensor_0018/kernel[497]#b6` | Yes |
| 7 | `tflite_tensor_0019/kernel[442]#b7` | Yes |

The historical prefix-selection rule and earlier conditional-damage knowledge
make this pool exploratory. It has no documented population inclusion
probabilities. Even a significant replication concerns these fixed masks,
not arbitrary KWS bits, optimal whole-network protection, other architectures,
or radiation-calibrated reliability.

## Exposure audit

`netsurf/data/kws_replication_exposure_v1.json` inventories 23 finalized
real-checkpoint KWS bundles found in canonical/implementation run roots and
implementation scratch engineering directories. Every inventoried scientific
payload hash was checked against its finalized inventory/manifest; all
conditional-bundle inventories were verified. Large M93 prediction inventories
were not rescanned. Exact test indices, partition hashes, bundle identities,
checksums-file hashes and observed seed metadata are retained. Embedded
historical selection partitions are conservatively treated as exposed too.

| Evidence | Previously used original test indices |
|---|---|
| M93, M95, M951, M952 evaluation | Within 0–999 |
| M953–M955 calibration / selection smokes | 1000–1003 / 2024–2027 |
| M953 final smokes | 3048–3053 |
| M954 final smokes | 3054–3059 |
| M955 final smokes | 3060–3065 |
| M956 engineering final runs and pinned tests | 3060–3061 |
| M956 selection pilot | 2024–2087 |
| M956 original final | 3066–3193 |

The union contains 1,214 test identities. **No inventoried fault-evaluation
identity intersects 3194–3449.** No new-window clips were loaded to choose the
plan or tune class balance. Their class composition and encoded input hashes
will be recorded at execution. Source order is TFDS interleaved blocks of 16;
contiguous windows are not automatically identical class distributions.

This is not a global exposure registry: failed/manual/out-of-root jobs and
concurrent evaluations cannot be certified absent. Run-time checks additionally
scan known run roots and the requested output root for finalized exposure
manifests and repeated replication seeds. Do not run concurrent confirmations
on the same window; disclose other exposure before execution. All clips were
already used in aggregate clean-reference validation and earlier label audits:
“unexposed” here means no recorded prior fault-outcome use, not inference-naive.

## Schema 6.5 and validation boundary

`schema_version="6.5.0"`, protocol
`independent-frozen-mask-replication/v6`, machine schema
`netsurf/schemas/conditional/v1_5.schema.json`.

The original schema-6.4 final-only loader is unchanged. The new loader verifies
the original artifact using its original final plan, checks the exact source
hashes and masks, then separately binds the new plan. Checkpoint/asset lock,
benchmark/backend, candidate ordering, source S5 context, cardinality, BER,
loss and original selection construction must match. Missing evidence is an
error, never permission to rerun selection.

Existing scientific fields remain: `[benchmark]`, `[samples]`, `[risk]`,
`[[candidates]]`, `[backgrounds]`, `[restricted_oracle]`,
`[protection_context]`, `[paired_swap_evaluation]`, `[execution]`, `[output]`.
Inactive calibration/selection fields retain the original provenance but are
not executed. The exact replication knobs are:

| Field | Supplied plan | Meaning |
|---|---|---|
| `samples.evaluation_role` | `conditional_final_evaluation` | Fixed original-test-index mapping |
| `samples.evaluation_offset` | 146 | 3048 + 146 = original index 3194 |
| `samples.evaluation_count` | 256 | Last original index 3449 |
| `paired_swap_evaluation.background_count` | 256 | Independent master fields, not inference count |
| `paired_swap_evaluation.seed` | 957901 | Field i seed = 957901 + i×1000003 |
| `paired_swap_evaluation.comparisons` | Existing ordered three names | Selected–S5 primary, two secondary contrasts |
| `execution.confidence` | 0.95 | Fixed two-sided inference |
| `workflow.mode` | `replication_only` | No selection, ranking or subcube enumeration |
| `workflow.purpose` | `confirmation` | Reject previously exposed sample windows |
| `workflow.candidate_provenance` | `exploratory_frozen_m953_pool` | Does not imply random pool selection |
| `workflow.selection_bundle` | Actual source above | Read-only provenance |
| `workflow.selection_payload_sha256` | Actual hash above | Original scientific identity |
| `workflow.selection_artifact_sha256` | Actual hash above | Original frozen-mask file identity |
| `workflow.record_per_input` | true | Opt-in loss contributions from the same inference calls |
| `workflow.replication_plan` | Worktree-local JSON below | Exclusive, read-only prespecification artifact |
| `workflow.replication_plan_sha256` | File hash below | Required at execution |

The new seed's **derived** field-seed set is disjoint from the original 32
selection seeds, 64 original final fields, and inventoried historical seed
metadata. Different seed integers identify independent pseudorandom streams
under the existing generator contract; that is not a mathematical proof of
PRNG independence. No random draws are made during plan generation.

Changing scientific configuration after plan freezing causes rejection. A
researcher may prepare a different prespecified plan before observing outcomes,
under a new name and exposure audit; never overwrite an old plan. A draft
without plan binding can be validated/frozen but cannot execute.

## Prespecified statistics and diagnostics

The scalar endpoint is the **mean expected categorical-cross-entropy
difference**, target minus baseline. Negative favors target. Every exact swap
path is summed inside a master field; the field is the independent unit.
The engine retains all candidate interaction orders and the previous XOR,
restoration, ideal suppression and paired schedule semantics.

- Primary selected−S5: two-sided 95% Student-t mean CI and unadjusted paired
  t-test p-value (equivalently a one-sample t-test of whole-field differences).
- Two secondary contrasts: marginal mean CIs/effects, Holm-adjusted p-values
  across those **two** hypotheses, plus 97.5% individual Bonferroni intervals
  giving at least 95% simultaneous secondary-family coverage under the
  component interval assumptions. The legacy three-contrast supplemental
  intervals remain identifiable in the underlying paired summary; the M957
  analysis/report uses the new prespecified primary/secondary distinction.
- Degenerate all-zero contrasts have p=1; nonzero zero-variance observations
  have an explicitly undefined p-value rather than invented finite certainty.
- Descriptive signed counts, linear empirical quantiles, median, mean−median,
  worst five adverse fields, and top absolute 1%/5%/10% contributions to the
  full mean (signed sums divided by the original n).
- Leave-one-field-out means describe influence only; **no fields are removed
  from any inferential estimate**, no alternate trimmed primary is selected.

Student-t inference is approximate for strongly nonnormal field effects.
Rare adverse tails can make a finite-sample mean unstable despite frequent
favorable backgrounds. Favorable-background probability and median risk are
different endpoints and cannot replace an inconclusive mean-risk result.

Optional `artifacts/paired_input_contributions.json` retains per-clip
`sum_steps p*(CE_removed-CE_added)` for each field/comparison, exact sample
identities/hashes, and the residual between its mean and the unchanged scalar
contrast. Capture uses the same TFLite calls and does not modify scalar
reductions. A 1e-12 **diagnostic reduction** check allows float summation-order
roundoff; it does not weaken any INT8/reference conformance gate. The primary
CI still conditions on fixed clips. The crossed field×input cells are not
65,536 independent replications; no input-sampling CI is claimed here.

## Cost and frozen plan

Researcher configuration:
`config/examples/m957_kws_frozen_replication_k8_v1.toml`.

Final plan: `workspace/plans/m957_kws_frozen_replication_k8_v1_diagnostics.json`.

- Exact file SHA-256: `fd39aa016b6308393b172722a5c8e6177734511a189bd3334ee72480e9e57189`.
- Canonical plan payload SHA-256: `8f8b617e54391136d7e8a7c10d23102080681c0ec22f797b232d3a9be4e6dce8`.
- Nominal bound: 256×3×2×2 = **3,072 state requests**, **786,432 clip
  inferences**, plus the 16-clip setup reference check.
- Each actual pair differs by one swap: **1,536 requested states / 393,216
  clips before cache reuse**. No selection or gradient calls occur.
- The old 64-field/128-clip final run measured 139.626 seconds and 192 unique
  authoritative states shared across its three contrasts. An input-linear
  eightfold extrapolation is about 18.6 minutes; allow roughly **15–25 CPU
  minutes** as a planning range, not a guarantee or power claim. Setup, shared
  caches and machine contention change this estimate.
- Per-input diagnostics add at most 196,608 field/input/contrast numbers
  (1.5 MiB raw float64, larger in JSON), with no extra inference. Exact event
  schedules remain saved and generally dominate bundle size.

An earlier metadata-only draft plan without per-input diagnostics is preserved
as `..._v1.json`; it was never executed on production inputs. Only the
`..._v1_diagnostics.json` artifact is bound by the delivered TOML.

## Exact researcher commands

Use tmux and the pinned interpreter; no installation, background wrapper or
environment migration is required:

```bash
cd /projects/netsurf/manuelbv/workspace/implementation-worktrees/m957
export NETSURF_WORKSPACE="$PWD/workspace"
/projects/kona-assets/manuelbv/.venv-mlperf/bin/python -m netsurf conditional validate \
  config/examples/m957_kws_frozen_replication_k8_v1.toml --resolved
```

After reviewing the plan and confirming no unrecorded exposure, run **once**:

```bash
/projects/kona-assets/manuelbv/.venv-mlperf/bin/python -m netsurf conditional run \
  config/examples/m957_kws_frozen_replication_k8_v1.toml
```

The existing exact S5 source bundle is required:
`/projects/netsurf/manuelbv/workspace/runs/m93_kws_static_ber_sweep1000_v1/20261029T183410.488907Z-32c31070`.
The locked Speech Commands dataset and INT8 model paths in the TOML must
remain accessible. TF2.21 is not an equivalent backend.

On a fresh checkout where the gitignored plan is absent, reconstruct it from
the TOML **before inference**, then compare the printed file hash above:

```bash
/projects/kona-assets/manuelbv/.venv-mlperf/bin/python -m netsurf conditional freeze-replication \
  config/examples/m957_kws_frozen_replication_k8_v1.toml \
  --output workspace/plans/m957_kws_frozen_replication_k8_v1_diagnostics.json
```

This deliberately fails if that destination already exists. Python API:
`load_conditional`, `replication_plan`, `freeze_replication_plan`,
`verify_replication_plan`, `execute_conditional_damage`, and
`create_conditional_bundle`. Plan freezing and validation require no TensorFlow
or model inference. SciPy is required for this protocol's t inference; no
normal approximation is silently substituted.

## Validation evidence

- Full canonical suite after the final numerical changes: **282 passed,
  7 skipped, 6 failed**. Four failures require an absent worktree-local MNIST
  checkpoint (validators also check the configured MNIST dataset); two are
  stale ResNet asset-lock/checkpoint expectations. All six were independently
  reproduced on unchanged M956 `9a96d5e`. No asset, test threshold, environment
  or checkpoint was changed to conceal them.
- Pinned TF2.14.1 suite covering M957, M956, M953, M955, conditional damage,
  restricted enumeration, and M93 KWS/ToyCar: **72 passed, 2 skipped**. Both
  skips are optional generic `jsonschema` checks; that package is absent.
  The executable strict validators and rejection tests passed. The user may
  install `jsonschema>=4,<5` if those extra checks are desired; no installation
  was performed.
- M957 fast suite: **15 passed, 2 skipped** (optional generic schema checker
  and pinned-TF integration in the canonical environment).
- Bounded real replication: three fields, two already exposed clips
  **3060–3061**, seed **957211**, exact original eight-bit pool and frozen masks.
  Repeated scientific payload:
  `5689cc2c2d9baaef4c910be12439a07c76b2f5b5b8fc8efba02cb83f4391eb46`.
  The guard rejects model-selection access, subcube enumeration and reserved
  final-window access. State restoration, global cardinality, proposed =
  applied + suppressed, scalar/per-input consistency and exact repeatability
  passed. Disabling diagnostic capture gave exactly identical scalar field
  and path records. No inference consumed 3194–3449.
- A replay of the old three-field/two-clip schema-6.4 engineering final
  reproduced its original scientific digest exactly:
  `5b4efe74ccf9b81f9e685b1203d88168d9a4ba7acb2054faedf12d857b0d3556`.
  The old bundle was not rewritten.
- Existing exact-enumeration tests retain sign/orientation, beneficial faults,
  higher-order interactions, all equal-budget masks and multi-step expectation
  telescoping. New tests cover hash corruption, altered context/candidates,
  final-plan edits, derived-seed collisions, exposure rejection, complete field
  pairing, heavy-tail influence, correct t intervals and primary/secondary
  inference. No p-value is inferred from marginal-CI overlap.

Sample engineering report:
`workspace/scratch/pytest-m957-pinned-complete/test_pinned_kws_replication_no0/runs/m957_engineering_replication/20261030T175108.112878Z-6827319a/report.md`.

```bash
/mnt/raid0/ml/conda_envs/manuelbv/envs/netsurf/bin/python -m pytest -q --tb=short
/projects/kona-assets/manuelbv/.venv-mlperf/bin/python -m pytest -q \
  tests/test_m957_replication.py tests/test_m956_preflight.py \
  tests/test_m953_same_distribution.py tests/test_m955_paired_swap.py \
  tests/test_m95_conditional_damage.py tests/test_m95_restricted_subcube.py \
  tests/test_m93_multibench.py
```

## ZIP handoff and Mac download

After the researcher finishes, set `M957_RUN` to the **exact** printed finalized
run directory. The archive includes both complete result bundles (new
replication and original selection), the TOML, frozen plan, ledger and handoff.
It does not distribute datasets or model binaries. Choose a new archive name
if that name already exists.

```bash
cd /projects/netsurf/manuelbv/workspace/implementation-worktrees/m957
M957_RUN="/ABSOLUTE/FINALIZED/REPLICATION/RUN"
M957_SELECTION="/projects/netsurf/manuelbv/workspace/implementation-worktrees/m956/workspace/runs/m956_kws_s5_selection_k8/20261030T165154.608989Z-b465f389"
M957_ARCHIVE="workspace/exports/m957_replication_evidence.zip"
mkdir -p workspace/exports
test -f "$M957_RUN/FINALIZED" && test ! -e "$M957_ARCHIVE" && \
  /projects/kona-assets/manuelbv/.venv-mlperf/bin/python -m zipfile -c "$M957_ARCHIVE" \
  "$M957_RUN" "$M957_SELECTION" \
  config/examples/m957_kws_frozen_replication_k8_v1.toml \
  workspace/plans/m957_kws_frozen_replication_k8_v1_diagnostics.json \
  netsurf/data/kws_replication_exposure_v1.json \
  docs/M957_FROZEN_MASK_REPLICATION.md
sha256sum "$M957_ARCHIVE"
```

On the Mac, replace `KONA_SSH_HOST` with the researcher's actual reachable SSH
host/alias (not an assumed hostname):

```bash
scp manuelbv@KONA_SSH_HOST:/projects/netsurf/manuelbv/workspace/implementation-worktrees/m957/workspace/exports/m957_replication_evidence.zip ~/Downloads/
shasum -a 256 ~/Downloads/m957_replication_evidence.zip
```

The package contains immutable scientific provenance, not portable copies of
the external dataset/checkpoint/S5 baseline dependencies. Do not edit old
bundles to relocate paths. Matching hashes verify byte identity, not immunity
to undisclosed prior exposure or scientific validity of population claims.
