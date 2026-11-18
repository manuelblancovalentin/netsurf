# M97 bounded validation receipts

All paths below are relative to
`<repo>/workspace/implementation-worktrees/m97`.
Implementation branch: `m97-fkeras-adapters`, base `4d646a83014bd7a0aed11045bf9f3456e341ee56`.
These are engineering checks, not a publication campaign or evidence of ranker
superiority. Original asset locks, historical worktrees, finalized bundles and
the canonical branch's researcher-owned configurations were not modified.

Implementation commits:

- `5c285218cdecf059257174ebe11a9de1d01fdbf8`: bounded continuous Hessian,
  paper/repository formulas, schema-3.5 integration, matched-budget examples.
- `ee71581284a277147b16b212ca7a7f2edb3a9988`: capability records/probes,
  streaming fixture, reference-source tests and ResNet lock/path audit.

Authorship and committer identity are the configured researcher identity.
The source-controlled documentation is committed separately. No merge or push.

## Completed suites

| Check | Result |
|---|---|
| Full canonical `python -m pytest -q -rs` | 315 passed, 11 skipped, 5 warnings; 93.34 s |
| Pinned MLPerf/M95–M97 regression subset (command below) | 115 passed, 4 skipped, 5 warnings; 244.43 s |
| Latest three real/source M97 integration tests | 3 passed; 30.13 s |
| Targeted capability/eigensolver/schema unit tests | 10 passed; 0.82 s |
| Final M97 integration/unit checks plus explicit optional-gate checks | 13 passed, 4 skipped; 30.59 s |
| Finite-difference HVP resource check | 1 passed; 28.87 s process elapsed, 499,892 KiB peak RSS |
| Two equivalent protection runs | Identical complete scientific payload digests |
| Finalized bundle readers | All three principal bundles' inventories/checksums verified |
| Saved-data plotting | Canonical Python regenerated reports/figures without ML inference |

The pinned subset contains `test_mlperf_tiny_adapter`, `test_m93_multibench`,
`test_m95_fkeras`, `test_m95_conditional_damage`, `test_m95_restricted_subcube`,
`test_m953_same_distribution`, `test_m955_paired_swap`, `test_m956_preflight`,
`test_m957_replication`, `test_m96_cross_pool`, and all three `test_m97_*` files.
Existing ResNet exact inference, companion, storage and fault tests pass in the
pinned environment after the documented path/lock-metadata reconciliation.

Canonical skips distinguish pinned TF 2.14.1/preprocessing requirements from
modern TF, three optional JSON Schema library tests, and the unavailable
**worktree-local** M96 campaign receipt. The historical receipt is not copied or
regenerated to remove that skip. Pinned skips are those three JSON Schema tests
and that optional local receipt, not failed reference inference. `jsonschema`
is absent in both tested interpreters. The strict executable Python validators
still run; the extra independent machine-schema checks require the user to run,
if desired:

```bash
<netsurf-env>/bin/python -m pip install 'jsonschema>=4.18,<5'
<assets>/.venv-mlperf/bin/python -m pip install 'jsonschema>=4.18,<5'
```

No installation was performed. Pinned Matplotlib is also absent; installed
canonical Matplotlib is the verified offline alternative. Deprecation warnings
from modern `tf.lite.Interpreter` and pinned Python audio/resource libraries
remain visible; the reference backend was not replaced to silence them.

## Finalized experiment identities

| Run | Relative directory | Scientific payload SHA-256 |
|---|---|---|
| Protection A | `workspace/runs/m97_kws_fkeras_protection_smoke/20261031T203537.286229Z-3fce5c82` | `2db342a2f4cb38cf553124cb10eba115d566255d84afec6b9e82d97a14e29b5e` |
| Protection B | `workspace/runs/m97_kws_fkeras_protection_smoke/20261031T203809.159386Z-1c47256d` | `2db342a2f4cb38cf553124cb10eba115d566255d84afec6b9e82d97a14e29b5e` |
| Protection C, final runtime-provenance check | `workspace/runs/m97_kws_fkeras_protection_smoke/20261031T204633.165614Z-ec595393` | `2db342a2f4cb38cf553124cb10eba115d566255d84afec6b9e82d97a14e29b5e` |
| Isolated-bit audit | `workspace/runs/m97_kws_fkeras_empirical_smoke/20261031T203744.905741Z-00ed6be5` | `ccdcace3393d8ac073e06ffe6d949ae1608b4e2d2508e9e73d05fe66e703bbc9` |

These runs precede the final source commit and correctly record a dirty source
fingerprint. Their saved numerical definitions/eigenvectors are included in the
payload. Absolute paths, elapsed timings and report regeneration are not part
of the scientific digest. Earlier M97 diagnostic attempts remain untouched;
one `.incomplete` attempt stopped at optional Matplotlib plotting and is **not**
counted as finalized evidence. Protection C records SciPy 1.17.1 explicitly in
its schema-3.5 execution manifest and preserves exactly the same scientific
digest. No earlier manifest was edited.

KWS checkpoint SHA-256:
`aeea436800704fce17b17292e4412630ad856e9d777c044c64ef748a880bd0ae`.
Rank/calibration: train indices 0–1. Empirical: validation indices 0–3.
Protection: test indices 0–3. Indices refer to pinned TFDS interleaved blocks
of 16; input identities and hashes are in each manifest. These engineering
inputs are already exposed, not the reserved confirmation windows.

### Actual allocation agreement

| Coverage | Protected bits | Paper vs repository symmetric difference | MSB vs paper | MSB vs repository |
|---|---:|---:|---:|---:|
| 6.25% | 11,008 | 1,802 | 8,368 | 8,416 |
| 12.5% | 22,016 | 0 | 0 | 0 |
| 18.75% | 33,024 | 1,802 | 8,368 | 8,416 |
| 25% | 44,032 | 0 | 0 | 0 |

The full-population raw FKeras score correlation is 0.9455215245, with
175,888 changed average ranks out of 176,128. These are repeated word scores,
not a global bit-policy ordering. Actual tier-first masks are the allocation
evidence. Partial tiers exchange 901 bits between the two variants; complete
tiers are exactly degenerate. This is a demonstrable implementation difference,
not a protection-effectiveness conclusion.

### Exact isolated-bit outcomes and score correspondence

The uniform without-replacement sample has 16/176,128 marginal inclusion
probability per bit. On four validation clips, eight flips increase CE, one
decreases it, seven produce zero CE change; signed changes range from
−0.00600613488 to +0.02230349990. None changes accuracy on this tiny sample.

| Numerical score | Spearman vs positive CE damage | Tie-aware top-5 recall | Construction s |
|---|---:|---:|---:|
| Corrected MSB | 0.506997 | 0.60 | structural ordering |
| S4 | 0.410670 | 0.60 | 0.353 |
| S5 | 0.410670 | 0.60 | 3.763 shared |
| S6 | −0.281091 | 0.40 | same shared S5/S6 computation |
| FKeras paper magnitude | 0.168359 | 0.60 | 1.114 shared |
| FKeras repository magnitude | 0.014161 | 0.40 | same shared eigensolve |

Random expected recall is 5/16=0.3125 on this candidate set. These noisy point
values are not a population comparison or an estimator-selection criterion.
Inherited within-observed-stratum bootstrap intervals condition on a sparse
stratum layout (many singleton strata); they are not trustworthy whole-network
uncertainty and are not interpreted as significance here. Raw score correlation
is distinct from the lexicographic FKeras ordering used for protection.

### Protection outcomes, costs and limits

The two fields contain 17 and 16 proposed events. Every policy uses the same
field identities and exact proposed/applied/suppressed accounting. Six methods
× four budgets plus zero/full controls give 52 evaluation records, **two**
independent fault fields. All accuracies are 1.0. Clean CE is 0.0048962188561;
faulted CE ranges from 0.0039138993211 to 0.0058785383911. Beneficial changes
are retained. Equal masks give equal results. This experiment cannot establish
a practical advantage for any method.

Protection A's execution manifest reports 62.36 s coordinator elapsed and
964.20 MiB process peak RSS. S4 takes 0.340 s; S5/S6 share 3.511 s; both FKeras
variants share 1.111 s and 25 HVP products. Source verification, sample loading,
full-population ordering, agreement tables and serialization dominate much of
the remaining time. The two principal bundles occupy about 165 MiB (protection)
and 135 MiB (empirical), including >1 million full-population estimator rows.
These are actual records, not millions of new fault trials. Large studies need
both inference and serialization/storage budgets. Estimates scale by input and
trial count only for the corresponding repeated operation, not for all setup.

## Capability probe receipts

Each `workspace/runs/<name>/probe.json` includes a separately hashed payload,
exact clean/faulted/restored outcomes and two proposed events. Unprotected:
2 applied, 0 suppressed. Full protection: 0 applied, 2 suppressed.

| Name | Stable payload hash |
|---|---|
| `m97_probe_kws_v1` | `6eedca39b4c83e6908b5b416afc26c1e0f14781709187f0dc27abeec7bbf70ae` |
| `m97_probe_toycar_v1` | `18445c306682b500beaf541b6e6cc2f129d078a479d051d4d83602ff08289c49` |
| `m97_probe_resnet8_v1` | `7300dbbf74ea199c700c390e00cdf1b51203612a19ea74418bcb08511fbe1aeb` |
| `m97_probe_streaming_v2` | `64ea8258a108507de683c99b1949bad4ae303e892eadcb786c24bf3f004fb781` |

The newest streaming probe includes the mandatory exact clean input/output gate:
load 2.284 s, whole-stream fault evaluation 0.157 s, total 4.905 s, peak RSS
566.29 MiB. These are not a full streaming benchmark result. The tiny KWS probe
is 2/2 correct; ResNet 1/2 correct; ToyCar AUC=1.0 across two files. Such tiny
outcomes neither replace nor contradict the archived full-reference quality
records; they validate execution and identity, not benchmark-wide quality.

## Derived figures and report destinations

- `workspace/runs/m97_derived_protection_v2/report.md`: actual mask equivalence,
  matched outcomes, trial distributions, coverage curves, layer/bit compositions.
- `workspace/runs/m97_derived_empirical_v2/report.md`: raw-score correspondence,
  exact measured damage, top-k recall, tensor/bit breakdowns and construction cost.

All figures come from saved measurements with canonical Python, no TFLite
reruns. Sources retain their original finalized checksums. See the source
documents for exact run and regeneration commands.

## Export and Mac retrieval

The prepared `workspace/exports/m97_handoff.tar.gz` contains the changed-source
archive, full principal finalized protection/empirical bundles, the equivalent
repeat run's manifest/checksums, probe receipts, derived reports/figures and
bounded test logs. It excludes datasets, model files, private environments,
external repositories, paper PDFs, the incomplete attempt and unrelated runs.
The changed-source archive is an overlay on the base commit, not a standalone
clone. Preserve original relative paths when inspecting bundles.

To make an alternative complete evidence archive (new destination only):

```bash
cd <repo>/workspace/implementation-worktrees/m97
tar -czf workspace/exports/m97_evidence_copy.tar.gz \
  docs/M97_FKERAS_REPRODUCTION.md docs/M97_BENCHMARK_ADAPTER_STATUS.md docs/M97_VALIDATION_RECEIPTS.md \
  config/examples/m97_kws_fkeras_protection_smoke.toml config/examples/m97_kws_fkeras_empirical_smoke.toml \
  workspace/runs/m97_kws_fkeras_protection_smoke/20261031T203537.286229Z-3fce5c82 \
  workspace/runs/m97_kws_fkeras_empirical_smoke/20261031T203744.905741Z-00ed6be5 \
  workspace/runs/m97_derived_protection_v2 workspace/runs/m97_derived_empirical_v2
sha256sum workspace/exports/m97_evidence_copy.tar.gz
```

On the Mac, using the researcher's existing SSH alias (replace `kona` if needed):

```bash
scp kona:<repo>/workspace/implementation-worktrees/m97/workspace/exports/m97_handoff.tar.gz .
scp kona:<repo>/workspace/implementation-worktrees/m97/workspace/exports/m97_handoff.tar.gz.sha256 .
shasum -a 256 -c m97_handoff.tar.gz.sha256
mkdir m97_handoff
tar -xzf m97_handoff.tar.gz -C m97_handoff
```

No remote upload or push is performed by these instructions or this milestone.
