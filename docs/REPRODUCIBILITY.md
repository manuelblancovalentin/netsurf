# Reproducibility

## Modern M2 environment construction

NetSuRF M2 is validated on Python 3.11.17. `proj` selects the Kona environment
but is not required by the software and is not a dependency lock.

On Kona with `proj`:

```bash
cd /home/manuelbv/netsurf
proj activate netsurf
python -m pip install -r dependencies/modern-ml-direct.txt
python -m pip install --no-deps -e .
python -m pip check
netsurf doctor
python -m pytest -q
```

## Deterministic CPU multiprocessing

The one-worker command remains the reference. Configurations with independent
replications or search conditions can use:

```bash
netsurf run config/experiments/study.toml --workers 8 --threads-per-worker 1
netsurf batch a.toml b.toml --workers 2 --threads-per-worker 2
netsurf pruning batch prune-a.toml prune-b.toml --workers 2 --threads-per-worker 2
```

Serial and parallel executions create distinct immutable run directories;
compare their `scientific_payload_sha256` values. See
`docs/PARALLEL_EXECUTION.md` for boundaries, thread controls, provenance,
failure semantics, and worker-sizing guidance.

On another server without `proj`:

```bash
git clone <approved-netsurf-remote> netsurf
cd netsurf
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r dependencies/modern-ml-direct.txt
python -m pip install --no-deps -e .
python -m pip check
netsurf doctor
python -m pytest -q
```

The user controls installation and site modules. These commands are a recorded
recipe, not something NetSuRF runs automatically. Set `NETSURF_DATASET_ROOT`
when the shared dataset tree is not `/ml/datasets`; set `NETSURF_WORKSPACE` to
change the generated-work root. The normal destination remains the ignored
repository-local `workspace/` tree. Archive `python -m pip freeze` with a
deployment record when exact transitive recreation matters. No EDA tool or
`proj` module is required for software experiments.

CPU-only TensorFlow messages saying that CUDA drivers are absent are expected.
The adapter fixes `TF_ENABLE_ONEDNN_OPTS=0`, `TF_DETERMINISTIC_OPS=1`, and
`CUDA_VISIBLE_DEVICES=-1` before TensorFlow import; importing TensorFlow first
under conflicting settings causes a clear failure. See
`dependencies/MODERN_ML.md` for the evidence and HLS-library boundary.

## Modern diagnostics and reference benchmark

```bash
netsurf doctor
netsurf benchmarks list
netsurf benchmarks info numpy_reference
netsurf benchmarks info mnist_quantized
netsurf validate config/experiments/numpy_reference.toml --resolved
netsurf run config/experiments/numpy_reference.toml
```

The fixed NumPy 2-3-2 network has 17 exactly quantized parameters (102 encoded
bits) and eight packaged samples. Tests exhaustively inject all 102 bits,
compare analytic gradients with finite differences, exercise multi-bit XOR and
per-inference replay, and verify state restoration. It is a scientific
reference, not a publication benchmark.

## Modern MNIST preparation

Prerequisite dataset:

- `${NETSURF_DATASET_ROOT:-/ml/datasets}/mnist/mnist.npz`
- SHA-256 `731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1`

Prepare once:

```bash
TF_CPP_MIN_LOG_LEVEL=2 CUDA_VISIBLE_DEVICES=-1 TF_DETERMINISTIC_OPS=1 \
  netsurf benchmarks prepare mnist_quantized --epochs 3 --train-samples 20000
```

This trains on `x_train[0:20000]`, reserves `x_train[50000:60000]` for
validation/ranking, evaluates `x_test`, quantizes each kernel and bias to
NetSuRF `q<6,2,1>`, and atomically writes a read-only checkpoint to
`workspace/benchmarks/mnist_quantized/checkpoint.npz`. It refuses overwrite.
Expected scale is CPU-only, three short epochs, approximately one minute, with
no HLS or synthesis.

The checkpoint prepared on 2026-10-22 has SHA-256
`1c58daf1bb23d1cdbaf38e68e4d957145be62435fd379bc3716904d57aa77780`.
Exact quantized inference on the first 256 test samples is 0.92578125 accuracy
and 0.2608608603477478 categorical cross-entropy. Training may not reproduce
byte-identically across CPUs or TensorFlow builds. Once prepared, the
checkpoint and checksum are the stable scientific input.

## Modern single- and multi-ranker runs

```bash
netsurf validate config/experiments/mnist_msb.toml
netsurf run config/experiments/mnist_msb.toml
netsurf validate config/experiments/mnist_ranker_comparison.toml
netsurf run config/experiments/mnist_ranker_comparison.toml
```

Both use 256 evaluation images, 256 disjoint ranking images, 12 independent
persistent-fault trials, logical BER 0.001, and 10% equal-bit protection. The
comparison uses matched schedules for MSB-first, seeded random, first-order
gradient-times-exact-delta, no-protection, and full-protection controls. Typical
runtime is under one minute on a Kona CPU and output is below 20 MB per bundle.
No training occurs.

Expected scientific payload SHA-256 values for the validated checkpoint are:

- single MSB: `696cc0c57ba6107c3fd687072cf400e7d69bc9555d1ce57dcd00f2d621ca8c58`
- comparison: `f0fa9c19929b577fb8d47f2d479a8730b60606182bc564f00d3a07bf2ce8a32a`

The explicit twice-repeated check is `python -m tests.integration_modern_mnist`.
It does not train; it verifies the checkpoint and two finalized comparison
bundles. Wall-clock ranker timings, timestamps, paths, and Git dirty metadata
are excluded from the digest.

Every run prints its unique directory. Inspect and regenerate presentation
without rerunning inference:

```bash
netsurf inspect workspace/runs/<experiment>/<run-id>
netsurf report workspace/runs/<experiment>/<run-id> \
  --output workspace/scratch/reports/<run-id> --plots
```

## M3 empirical susceptibility and composable policies

The following commands use the existing prepared checkpoint and never train:

```bash
netsurf validate config/experiments/numpy_single_bit.toml --resolved
netsurf run config/experiments/numpy_single_bit.toml

netsurf validate config/experiments/mnist_single_bit.toml
netsurf run config/experiments/mnist_single_bit.toml

netsurf validate config/experiments/mnist_msb_policies.toml
netsurf run config/experiments/mnist_msb_policies.toml
```

The NumPy command exhausts all 102 bits and took under one second of scientific
execution on Kona. Two independent finalized runs produced the identical
scientific payload SHA-256
`65e00c15cc833e72845525d4ebcd12781c1400befee2de8144c2339704193489`.
Its validated overall Spearman correlations with exact positive loss damage
were 0.342 for ordinary first order, 0.481 for per-example first-order
magnitude, 0.471 for the exact diagonal second-order approximation, and 0.434
for the 32-probe Hutchinson approximation. These are results for this 102-bit
reference only.

The bounded MNIST empirical template evaluates 96 candidates stratified by
layer, parameter class, bit position, and first-order score quantile. It uses
64 disjoint ranking samples and 64 disjoint empirical-validation samples. The
validated run took about 21 seconds end to end and produced payload SHA-256
`635df014d4468a4ebd5b316a87064a3aa78b11357c29e4e26894ff1ab5bbd454`.
The output bundle is approximately 102 MB because it records all 152,700 scores
for each configured estimator. The resulting correlations and top-k measures
are explicitly sample-limited, not whole-network claims.

The bounded MSB policy template uses six paired persistent-fault trials, 128
final-evaluation samples, 10% and 1/6 coverages, two independent random ranking
masks, and matched fault schedules. It took about 24 seconds and produced
payload SHA-256
`c4fb13979250e7d3759b9d751675c652d80a6c48c467b207e72f7822320f7f37`.
At 1/6 coverage, every MSB-primary variant produced the identical 25,450-bit
mask SHA-256
`d2b98a858f2799840a697ac3c390df0cfe3fb31e24147842fe67e2ff29191045`;
at 10%, tie-breaking changed the masks and observed outcomes.

The more demanding user-run templates are:

```bash
netsurf run config/experiments/mnist_estimator_comparison.toml
netsurf run config/experiments/mnist_multi_coverage.toml
```

The former raises the empirical sample to 192 candidates, ranking/empirical
sample counts to 128, and Hutchinson probes to eight. A conservative estimate
from the 96-candidate run is 30–60 seconds and roughly 120 MB on Kona CPU. The
latter evaluates three policies at three coverages across 12 paired trials; a
conservative estimate from the six-trial policy run is under two minutes and
roughly 50 MB. These estimates are machine- and filesystem-dependent. Neither
command is an exhaustive 152,700-bit susceptibility campaign.

## Offline cross-benchmark mechanism audit

The mechanism audit reads finalized bundles only. It does not import
TensorFlow, rerun inference, or alter source manifests:

```bash
netsurf mechanism-audit \
  --benchmark resnet8 /path/to/estimator-bundle /path/to/protection-bundle \
  --benchmark kws /path/to/estimator-bundle /path/to/protection-bundle \
  --benchmark toycar /path/to/estimator-bundle /path/to/protection-bundle \
  --output workspace/mechanism-audit/cross_benchmark_v1
```

Used source files are checked against their finalized bundle checksums. Score
CSVs and mask JSON are streamed, so multi-gigabyte artifacts are not loaded in
memory. The output is a separate checksummed derived analysis and must not be
presented as a new scientific experiment or causal mechanism. See
`docs/M93_CROSS_BENCHMARK_MECHANISMS.md`.

## M3.1 researcher-driven estimator studies

Validate before launching:

```bash
netsurf doctor
netsurf validate config/experiments/numpy_qpolar_cancellation.toml --resolved
netsurf validate config/experiments/mnist_estimator_uniform.toml --resolved
netsurf validate config/experiments/mnist_estimator_replicated.toml --resolved
netsurf validate config/experiments/mnist_multi_coverage_ber.toml --resolved
```

Run only the desired protocol:

```bash
netsurf run config/experiments/numpy_qpolar_cancellation.toml
netsurf run config/experiments/mnist_estimator_uniform.toml
netsurf run config/experiments/mnist_estimator_replicated.toml
netsurf run config/experiments/mnist_multi_coverage_ber.toml
```

The controlled NumPy experiment exhausts all 102 bits. On Kona its three
qPolarGrad estimator calls took about 0.003 seconds in total; the full CLI run,
including bundle plots and checksums, completed in about two seconds. Its
scientific payload SHA-256 is
`56c509f8f237a9833444c9ec1d94ea7db3ad052fec464be0514a6acedc9e68db`.
For the signed profile, clean signed-mean versus magnitude-before-mean scores
had Spearman 0.831830; 93 of 102 average ranks changed and top-10 overlap was
9. The matched nonnegative profile produced identical scores and order. These
are controlled-reference observations, not evidence of MNIST protection
superiority.

The uniform MNIST template uses 96 exact injected candidates and is expected to
be comparable to the validated 96-candidate M3 run: tens of seconds for
scientific execution and roughly 100–150 MB because full-network score tables
are retained. The replicated template uses 3 × 48 exact candidates and 2-, 4-,
and 8-probe Hessian variants per replication. Empirical injection work scales
approximately with candidate count times empirical sample count; Hutchinson
work scales with total probes times ranking sample count. From the bounded M3
measurements, reserve several CPU minutes and roughly 300–500 MB for that
template, then refine the estimate from its first replication. The multi-BER
template evaluates three times the logical-BER conditions of the corresponding
single-BER design; reserve up to several minutes and approximately 150 MB.
These are planning estimates, not performance guarantees.

Candidate count, sample counts, Hutchinson probes, replication count/strides,
coverage, logical-BER conditions, and trial count are all TOML fields. Each run
prints its unique immutable directory. `replication_aggregate.csv` reports
candidate/sample replication dispersion separately from per-probe diagnostics
and fault-trial statistics.

Interactive `netsurf run` commands show a compact Rich live view on stderr.
Redirected execution remains quiet by default; `--progress` forces periodic
plain-text lines without ANSI escapes, and `--no-progress` disables reporting.
The final result-directory JSON remains on stdout. Progress counts completed
coarse operations: shared schedule creation, actual mask evaluations, empirical
bit injections, and declared setup/analysis stages. The displayed ETA is for
the complete declared run, uses a monotonic clock and operation-specific
observed rates, and remains `Estimating...` until enough remaining work is
characterized.

Schema-3.1 protection reporting pools every configured replication by logical
BER, coverage, and policy. Pairing uses the exact `(replication, logical BER,
trial, schedule SHA-256)` identity. `protection_results.csv` reports pooled
fault-realization statistics, while `replication_protection_results.csv`
preserves within-replication summaries and `paired_differences.csv` retains raw
matched effects. These intervals remain conditional on the recorded model,
dataset, and masks; repeated fault schedules under one mask are not independent
ranking-mask samples.

To derive corrected presentation artifacts from an older finalized bundle
without modifying it:

```bash
netsurf report workspace/runs/<experiment>/<run-id> \
  --output workspace/scratch/derived/<run-id> --plots
```

The destination contains `derived_summary.json`, derived measurement tables,
plots, and `report.md`. The source bundle and scientific payload hash remain
unchanged. A replicated archive lacking explicit replication identities is
rejected instead of guessed.

The maintained wide stress configuration is ready for researcher execution:

```bash
netsurf validate config/experiments/mnist_policy_ber_wide_v1.toml
netsurf run config/experiments/mnist_policy_ber_wide_v1.toml
```

It declares 800 fault schedules and 11,200 mask/control evaluations across five
replications (12,037 total progress operations including setup and
finalization). BER values through 0.1 are logical stress conditions, not a
physical SEU-rate model. Relative to the two-BER source design,
fault-evaluation work is approximately doubled; actual runtime is refined live
from completed operations.

CPU-only TensorFlow's CUDA-driver and `cuInit` messages are benign. NetSuRF's
own lazy import suppresses this TensorFlow 2.21 pre-absl chatter by default;
`NETSURF_TF_IMPORT_QUIET=0` restores it for debugging. Direct user imports of
TensorFlow occur outside that wrapper and may still print it. NumPy is not the
source of these messages. An actual Python exception or changed payload is a
failure; use a fresh process when changing deterministic environment settings.

Without `proj`, invoke the active environment's `python -m netsurf` or the
installed `netsurf` entry point after following the Python 3.11 environment
recipe above. Do not copy Kona's absolute environment path to another server.

## Historical M1.5 compatibility

### Canonical environment and installation

The private `proj` project `netsurf` points to this repository and the dedicated
Python 3.11.17 environment at
`/mnt/raid0/ml/conda_envs/manuelbv/envs/netsurf`. Proj selects an environment;
it is not a dependency lock.

```bash
/opt/proj/bin/proj info --env netsurf
proj activate netsurf
python -m pip install --no-deps -e .
python -m tests.run
```

Expected M1.5 core result: 30 passed, 0 failed. From outside the checkout,
`python -c 'import netsurf; print(netsurf.__version__)'` must print `2.0.0`.

On Kona, interactive shell startup must initialize Lmod, Conda, and the proj
shell adapter. The repository does not modify those site shell settings.

### Storage prerequisites

- Dataset: `/ml/datasets/mnist/mnist.npz`, SHA-256
  `731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1`.
- Checkpoint under `/ml/experiments/netsurf/legacy/benchmarks/`, SHA-256
  `ba48aee8adbe7d04092d74f41958ec99112ae3e5f88ed56552ea9d1c4dba3006`.
- Output root: `<repo>/workspace/runs/`.

The generated-work root can be changed with `NETSURF_WORKSPACE`; the normal
and documented destination remains repository-local. Historical external runs
are not moved. Exact input paths and migration evidence are in
`STORAGE_MIGRATION.md`.

### First scientific milestone

The exact historical compatibility stack is TensorFlow 2.11.1 on Python 3.10,
preserved as `netsurf-legacy`. TensorFlow 2.11 has no Python 3.11 wheel, so the
milestone must remain in that environment until a newer stack is independently
shown to reproduce the same payload. The canonical core itself is validated in
the Python 3.11 `netsurf` environment.

First validate in the canonical environment:

```bash
conda run -n netsurf python -m tests.run
conda run -n netsurf python -m netsurf validate config/experiments/template.toml --resolved
```

The core suite does not import TensorFlow. The full template is then executed
in the preserved `netsurf-legacy` environment because that is where the
validated TensorFlow 2.11 checkpoint reader exists:

```bash
conda run -n netsurf-legacy python -m netsurf run config/experiments/template.toml
conda run -n netsurf-legacy python -m netsurf run config/experiments/template.toml
```

The optional automated form of the same twice-repeated check is:

```bash
conda run -n netsurf-legacy python -m tests.integration_mnist
```

Run `netsurf inspect <printed-run-directory>` for each output and compare the
reported scientific payload digests. This bounded command is CPU-only,
evaluates 256 held-out MNIST test images,
derives rankings from the same first 256 test images, runs 12 paired persistent-
fault trials at logical BER 0.001 and 10% equal-bit coverage. Each invocation
gets a unique run directory; no finalized bundle is overwritten.

Expected resource scale: CPU only, under one minute on the validated Kona run,
roughly 96 faulted model evaluations plus
ranking/gradient passes, and less than 10 MB of new result data. No training,
HLS, synthesis, or historical-result modification occurs.

Expected scientific payload SHA-256:
`07f4e88d7a36663fb120fafb97c28b80bab5dd3be13b91700755a4784b30bc1b`.

Each finalized run contains the exact TOML and resolved JSON configuration,
manifest and summary, scientific payload, full CSV measurements, exact masks
and schedules, checksums, report, supported plots, and `FINALIZED`. Exact
artifacts are write-once. Timestamps, source status, and run paths are excluded
from the scientific payload digest.

Reports can be regenerated without TensorFlow or inference:

```bash
conda run -n netsurf python -m netsurf report <run-directory> \
  --output workspace/scratch/reports/<run-id>
```

The modern canonical environment now includes Matplotlib 3.11.2. A core-only
installation can still regenerate Markdown without it; `--plots` requires the
declared `matplotlib>=3.8,<4` optional dependency.

## M4 conditional-search workflow

Start with the exact small reference and bounded prepared-checkpoint validation;
neither command trains a model.

```bash
proj activate netsurf
netsurf doctor
netsurf validate config/experiments/numpy_pgcs_exact.toml --resolved
netsurf run config/experiments/numpy_pgcs_exact.toml
netsurf run config/experiments/mnist_incremental_validation.toml
```

Keep a researcher-sized SSH run reconnectable:

```bash
tmux new -s netsurf-m4
proj activate netsurf
netsurf run config/experiments/mnist_static_guided_search.toml
# detach: Ctrl-b d; reconnect: tmux attach -t netsurf-m4
```

The polar comparison is `netsurf run
config/experiments/mnist_polar_guided_search.toml`. Inspect and reduce its
pair/context/iteration/replication budgets when turnaround matters. Redirected
runs retain ANSI-free progress on stderr and final JSON on stdout.

Without `proj`, activate a Python 3.11 environment, install the declared core
and research dependencies plus this checkout in editable mode, and use the same
commands. Point the TOML at immutable local dataset/checkpoint copies; every
manifest records their hashes. Do not copy Kona's absolute environment path.

## M4.1 controlled conditional-risk workflow

```bash
proj activate netsurf
netsurf validate config/experiments/numpy_pgcs_conditional_validation.toml --resolved
netsurf run config/experiments/numpy_pgcs_conditional_validation.toml
netsurf validate config/experiments/mnist_pgcs_matched_smoke.toml
netsurf run config/experiments/mnist_pgcs_matched_smoke.toml
```

The researcher-scale study is not an automated regression:

```bash
tmux new -s netsurf-m41
proj activate netsurf
netsurf validate config/experiments/mnist_pgcs_matched_research.toml --resolved
netsurf run config/experiments/mnist_pgcs_matched_research.toml
# detach: Ctrl-b d; reconnect: tmux attach -t netsurf-m41
```

Defaults use two logical BERs, five replications, eight methods, 20 natural
fields, and 24 conditional fields per method/BER/replication. Do not adapt
confirmatory allocations after inspecting held-out results.

## M4.2 topology-aware workflow

Run the algebraic reference first, then the prepared-checkpoint validation and
smoke study. None retrains the model.

```bash
proj activate netsurf
netsurf validate config/experiments/numpy_topology_polar_exact.toml --resolved
netsurf run config/experiments/numpy_topology_polar_exact.toml
netsurf validate config/experiments/mnist_topology_exact_validation.toml --resolved
netsurf run config/experiments/mnist_topology_exact_validation.toml
netsurf run config/experiments/mnist_topology_polar_smoke.toml
```

The larger study is researcher-run only:

```bash
tmux new -s netsurf-m42
proj activate netsurf
netsurf validate config/experiments/mnist_topology_polar_research.toml --resolved
netsurf run config/experiments/mnist_topology_polar_research.toml
```

Schema 4.2 records proposal-distribution shifts separately from exactly
verified swap benefits. Accurate pair interactions alone are not evidence that
an optimizer improves a protection mask. Logical BER 0.01 is a stress
condition, not a calibrated radiation rate.

## M4.3 mask-aware conditional-screening workflow

Run the exhaustive 102-bit reference first. It validates exact swap
decomposition, common-background candidate screening, full-population
factorized sums, dynamic updates, immutable outputs, saved-data plots, and the
exclusive timing audit.

```bash
proj activate netsurf
netsurf validate config/experiments/numpy_conditional_screening_exact.toml --resolved
netsurf run config/experiments/numpy_conditional_screening_exact.toml
```

Then use the prepared MNIST checkpoint for the bounded diagnostic and smoke
study. Neither command trains or overwrites the checkpoint.

```bash
netsurf validate config/experiments/mnist_factorized_aggregate_validation.toml --resolved
netsurf run config/experiments/mnist_factorized_aggregate_validation.toml
netsurf validate config/experiments/mnist_conditional_screening_smoke.toml --resolved
netsurf run config/experiments/mnist_conditional_screening_smoke.toml
```

The 25%-coverage diagnostic uses 32/16/32 ranking/search/final samples, one
search iteration, four final contexts, and explicit full-versus-restricted
aggregate populations. The smoke configuration uses 32/16/32 samples, two
iterations, four final contexts, and all principal M4.3 proposal paths. These
are bounded diagnostics, not publication experiments.

The researcher-sized comparison is deliberately not an automated regression:

```bash
tmux new -s netsurf-m43
proj activate netsurf
netsurf validate config/experiments/mnist_conditional_screening_research.toml --resolved
netsurf run config/experiments/mnist_conditional_screening_research.toml
```

It declares two budgets, two logical BERs, three replications, nine methods,
eight search iterations, and separate 32-context natural and conditional final
evaluations. Inspect disk/runtime after one bounded condition before committing
to the full matrix. Screening work scales approximately with replications ×
budgets × BERs × iterations × backgrounds × shortlisted candidates; exact
verification scales with the verified swap shortlist. Full-population
factorization setup additionally scales with the 152,700-bit eligible
population but its per-query aggregate is constant-size for the explicit dense
layers.

`candidate_screening_contexts.csv` preserves common-field paired contributions;
`factorized_guidance.csv` records coverage, drift, and timing attribution; and
`timing_phases.csv` is the only additive wall-time table. See
`docs/M4_3_AUDIT.md` for the M4.2 25%-coverage evidence and timing semantics.

## M4.4 controlled ablation workflow

M4.4 intentionally provides schema support rather than a maintained research
configuration. Author the study TOML separately using the exact field contract
in `docs/M4_4_AUDIT.md`, then run:

```bash
proj activate netsurf
netsurf validate /path/to/research-authored-m4_4.toml --resolved
netsurf run /path/to/research-authored-m4_4.toml
```

Start with one NumPy or bounded prepared-checkpoint replication and inspect
`controlled_proposals.csv`, `factorized_guidance.csv`, cache keys, background
schedule hashes, and exact-verification counts before increasing iterations,
BERs, budgets, or replications. Do not use held-out conditional results to
retune a confirmatory search within the same study.

## Sequential experiment suites

Existing configurations can share one hierarchical progress display without
changing any configuration or scientific protocol:

```bash
proj activate netsurf
netsurf batch config/experiments/first.toml config/experiments/second.toml
```

The queue is sequential: each experiment still creates its own immutable run
directory and scientific payload. Expected failures remain marked and later
queue entries continue; a nonzero exit status indicates that at least one entry
failed. Use `--progress` for occasional ANSI-free records when stderr is
redirected, or `--no-progress` for automation that consumes only the JSON
summary on stdout.

## M5 structured-pruning checkpoint preparation

M5 provides a schema and CLI, not a maintained research configuration or an
automatic training campaign. Author a schema-5 TOML using `docs/PRUNING.md`,
then validate before permitting any configured fine-tuning:

```bash
proj activate netsurf
netsurf pruning validate /path/to/research-authored-pruning.toml --resolved
netsurf pruning run /path/to/research-authored-pruning.toml
netsurf pruning inspect workspace/benchmarks/my_pruned_model/checkpoint.npz
```

With `fine_tuning.enabled = false`, preparation performs calibration,
importance calculation, physical tensor removal, exact six-bit requantization,
and immutable serialization only. Training cost when enabled scales with
`stages × epochs_per_stage × training_samples / batch_size`; reliability-aware
fine-tuning performs both clean and fault-augmented forwards per batch.

## Recovered MNIST-lite exact representation

Inspect the side-effect-free identity and accepted population before authoring
a configuration:

```bash
proj activate netsurf
netsurf benchmarks identity mnist_lite_exact_kb_v1
netsurf benchmarks info mnist_lite_exact_kb_v1
```

Use adapter `mnist_lite_exact_kb_v1`, quantization backend
`netsurf_bitcodec_v1`, and exactly
`injectable_parameters = ["kernels", "biases"]`. Do not use schema-4
Dense32-specific search configurations. The source checkpoint must be loaded in
the preserved compatible TensorFlow/QKeras environment; conversion is
deterministic and in memory, and the source remains unchanged. Full evidence,
digests, configuration fields, and blocked benchmark candidates are in
`docs/BENCHMARK_EXPANSION.md`.

## MLPerf Tiny ResNet-8 exact-storage adapter

The modern adapter has no dependency on deprecated NetSuRF, QKeras, or
`tensorflow_model_optimization`:

```bash
proj activate netsurf
netsurf benchmarks info mlperf_tiny_ic_resnet8_int8_v1
netsurf benchmarks identity mlperf_tiny_ic_resnet8_int8_v1
```

The official TFLite model, CIFAR-10 Python archive, official 200-image index,
and label identities are present through a verified neutral Kona Assets lock.
Use `asset_lock`, `inference_backend = "tensorflow_lite_builtin_ref_v1"`, and
an explicit `injectable_parameters` population. The registered output identity
requires TensorFlow Lite 2.14.1 `BUILTIN_REF`, no delegates, one thread;
TensorFlow 2.21 preserves clean accuracy but is not byte-equivalent. Exact
fields, output digests, storage counts, blocked estimators and commands are in
`docs/MLPERF_TINY.md`.
## M9 bounded MLPerf Tiny static-estimator gate

Run M9 only with the registered TensorFlow 2.14.1 reference environment:

```bash
/projects/kona-assets/manuelbv/.venv-mlperf/bin/python -m netsurf validate \
  config/experiments/m9_resnet8_static_estimator_validation_v1.toml --resolved
/projects/kona-assets/manuelbv/.venv-mlperf/bin/python -m netsurf run \
  config/experiments/m9_resnet8_static_estimator_validation_v1.toml
/projects/kona-assets/manuelbv/.venv-mlperf/bin/python -m netsurf run \
  config/experiments/m9_resnet8_static_protection_smoke_v1.toml
```

These are bounded validation commands, not publication campaigns. Do not
replace the reference runtime with the canonical TensorFlow 2.21 runtime.
See `docs/MLPERF_TINY_M9_STATIC_ESTIMATORS.md` for the mapping gate,
provisional acceptance criteria, measured resource costs and interpretation
limits.

## M9.1 publication-scale static comparison

The full correspondence audit, prespecified candidate design, finalized
320-candidate payload and protection-pilot resource estimate are in
`docs/M91_PUBLICATION_EVIDENCE.md`. Validate and run only with the explicit
TensorFlow 2.14.1 interpreter recorded there. The 32-realization protection
pilot is intended for a researcher-controlled `tmux` session and is not part
of the bounded regression suite.
