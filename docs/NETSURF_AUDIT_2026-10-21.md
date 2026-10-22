# NetSuRF Repository Recovery, Scientific Audit, and Development Roadmap

**Audit date:** 2026-10-21  
**Repository:** `/home/manuelbv/netsurf` (resolved path `/mnt/raid0/asic/projects/NU/netsurf/manuelbv`)  
**Branch / commit:** `master` / `a95ae571fadf482eb53d4e43314759fc203f99f4`  
**Manuscript:** `TCAD-2026-0117.R1_Proof_hi.pdf`  
**Scope:** read-only forensic reconstruction plus lightweight disposable tests. No source, model, dataset, ranking, result, HLS workspace, or synthesis artifact was changed.

## Evidence policy

- **Confirmed working:** executed during this audit or directly backed by a complete, internally consistent artifact.
- **Present, untested:** code/tool/artifact exists, but execution was not attempted because it would be expensive, destructive, licensed, or scientifically premature.
- **Incompatible:** a lightweight execution reached a reproducible failure in the current environment.
- **Missing:** the expected artifact or dependency was not found.
- **Unknown:** evidence is insufficient to decide.

Historical files prove that some workflow ran, not that the current checkout reproduces it.

---

# A. Executive assessment

NetSuRF is a substantial but unreleased research prototype, not a currently reproducible package. Its strongest recoverable assets are a coherent custom quantized-model layer stack; many checkpoints, rankings, and Monte Carlo CSVs; a real Catapult-to-RTL-to-Cadence Genus flow for five register-mapped accelerators; generated register-level TMR cells and synthesis reports; and several ranking implementations including qPolarGrad and a power-iteration Hessian adaptation.

The central experiment cannot presently be regenerated from the current checkout and current `netsurf` environment. A disposable four-sample qPolarGrad test constructed a model and entered the ranker, but failed at `netsurf/core/ranking.py:1998`: the code assumes Keras metric objects while the model stores metric names as strings. An empty metric list fails at the same site for a second reason (an empty floating-point index array). Historical notebooks also contain saved errors, hard-coded obsolete paths, and manual HLS conversion logic.

Several scientific claims do not match the code or artifacts:

- Equation (3) gives the wrong sign-bit perturbation for the implemented two's-complement encoding.
- The configured/stored `bitwise_msb` baseline protects LSBs first, so the manuscript's MSB/FORTUNE comparison is not supported by those masks.
- Injection draws one transient weight-fault mask per software batch, not per inference, and therefore depends on `batch_size`.
- Default evaluation uses training data although figures claim test performance.
- Table IV budget conversions and its linearity statement are inconsistent with Table IV itself.
- Checked-in synthesis workspaces do not reproduce Table IV's six points.
- Manuscript TinyML-Anomaly (179,959 parameters) is not the currently ranked/synthesized 8,560-trainable-value model.

The repository is suitable for recovery and validation, not immediate feature development. Historical results should be quarantined by provenance and regenerated only after minimal compatibility repairs, deterministic tests, and approval of the split, fault model, estimator, and benchmark manifests.

## Current capability matrix

| Capability | Status | Evidence / limitation |
|---|---|---|
| Import package | Present but side-effecting | Import initializes Nodus and creates config/DB/log state (`netsurf/__init__.py:70-74`). |
| Quantization codec/deltas | Confirmed for `q<6,2,1>` | Exhaustive 64-code / 384-bit-flip test: zero mismatches. |
| Build tiny custom FNN | Confirmed | Executed in existing `netsurf` environment. |
| Generate qPolarGrad | Incompatible | Reproducible failure at `ranking.py:1998` under TF/Keras 2.11. |
| Historical rankings/results | Present, not fully tested | 31 current ranking/result CSV pairs; 247 of each under `benchmarks_old`. |
| Ranking mask/injection | Code present; full smoke blocked | Statically traced; qPG failure prevented end-to-end execution. |
| GUI | Present, untested | Imported by package; no evidence it is canonical. |
| Notebooks | Historical, noncanonical | Stale paths, saved errors, manual conversion. |
| HLS | Artifacts present, untested | Catapult-oriented hls4ml outputs and RTL for five benchmarks. |
| Logic synthesis | Confirmed historical run | Genus DBs, netlists, timing/area/power reports, TMR wrappers. |
| Place-and-route/signoff | Missing from recovered outputs | No design GDS/DEF/SPEF or Calibre DRC/LVS result found. |
| Manuscript reproduction | Not currently possible | Configuration, provenance, executable, and scientific mismatches. |

---

# B. Architecture map

## Entry points

The current intended CLI call graph is:

```text
python -m netsurf
  -> netsurf.args.parse_arguments()
  -> netsurf.__main__.run_experiment()
  -> get_benchmark()
  -> get_training_session()
  -> run_injection_experiment()
  -> Experiment.rank()
  -> Experiment.run_experiment()
```

Symbols: `netsurf/__main__.py:25`, `:115`, `:193`, entry block `:261`; parser `netsurf/args.py:12`.

`netsurf/scripts/run_experiment.py` and `.sh` are obsolete: they hard-code `/home/manuelbv/WSBMR/workspace/dev`, old `bits_config`/`tmr_range`/`rad_range` names, and exports no longer present. Preserve them as provenance, not runnable entry points.

Notebook evidence:

| Notebook | Saved state | Conclusion |
|---|---:|---|
| `1_benchmarks_quantization_assertion.ipynb` | 0/9 code cells executed; `wsbmr` kernel | Template/evidence only. |
| `2_visualize_results.ipynb` | 5/12 executed | Old `q<6,0,1>` hierarchy. |
| `5_batch_running.ipynb` | 1/2 executed | Large report generator, not a maintained batch CLI. |
| `6_hls.ipynb` | 2/11 executed | Saved layer-count mismatch error; hard-coded paths. |
| `7_hls_complete.ipynb` | 9/25 executed | Saved `NameError`/`KeyError`; mixes analysis, HLS, RTL generation, and cost constants. |
| `plot_results.ipynb` | 3/34 executed | Historical plotting with old method names/hierarchy. |

## Actual data flow

1. `config/benchmarks.yaml` selects dataset/model/loss/metrics/session/pruning; `datasets.yaml` carries shapes. Python registries resolve strings.
2. `get_benchmark()` (`core/benchmarks.py:996`) creates `Benchmark` (`:153`), dataset/model, directory hierarchy (`:387`), and normally loads `.keras.latest` (`:529`). Construction is not read-only.
3. Training/session reload uses `__main__.py:25`; a second implementation exists at `core/benchmarks.py:1037`.
4. `Experiment` (`core/experiments.py:485`) owns ranker/results. `WeightRanker.build()` (`core/ranking.py:555`) selects a method. The base table (`:629-731`) enumerates every trainable variable and bit, including kernels, biases, learned alpha/beta, and batch-normalization parameters.
5. `QPolarGradWeightRanker` (`ranking.py:2129`) calls qPolar and gradient rankers, aligns original indices, and multiplies impact by absolute gradient (`:2139-2185`). It sorts by `pruned`, bit descending, then susceptibility (`:2108-2127`).
6. `BitFlipInjector.apply_ranking()` (`core/injection.py:533`) stores normalized per-variable/per-bit ranks in auxiliary `*_ranking` tensors; ranks at or below the protection threshold are perfectly protected.
7. `WeightLayer._inject()` (`dnn/layers.py:811`) draws Bernoulli masks, suppresses protected locations, sums bit deltas, and performs a transient perturbed forward pass without overwriting base weights.
8. `Experiment.run_experiment()` (`experiments.py:697`) Latin-hypercube-samples protection/BER, evaluates metrics, appends CSV rows, and stops when every 2-D bin reaches a count threshold.
9. Rankings are CSV plus header/config; results are CSV. `core/explorer.py` reconstructs a directory hierarchy. SQLite/Nodus tracks jobs, not primary results.

## Hidden coupling/state

- Import opens Nodus and mutates user state; benchmark construction creates metadata/directories.
- Layers carry delta, pruning, noise (`*_N`), rank, and counter variables.
- qPolar performs an all-bit attack; `_attack()` uses `assign_add`, so later scores can depend on prior calls unless reset.
- Result identity rests on compressed config hashes and directory conventions, not immutable versioned manifests.
- CLI methods are silently empty unless `--method`, `--method_suffix`, and `--method_kws` all appear with equal lengths. Keywords use `eval` (`args.py:70-83`, `ranking.py:575-600`).
- CLI always prefixes `pruned_<rate>_`; several current unpruned directories do not, so canonical loading misses them.

---

# C. Environment and benchmark inventory

## Environments

No requirements file, environment YAML, package metadata, or lockfile was found.

| Environment | Python | TensorFlow / Keras | QKeras | hls4ml | NumPy | Assessment |
|---|---:|---:|---:|---:|---:|---|
| `netsurf` | 3.10.16 | 2.11.1 / 2.11.0 | source link | `0.1.0.dev2562+g46b7a88` | 1.26.4 | Intended; qPG incompatible at metric handling. |
| `qkeras` | 3.10.13 | 2.11.0 / 2.11.0 | 0.9.0 | same dev build | 1.26.4 | Plausible historical analysis env; untested end to end. |
| `hls4ml` | 3.10.14 | 2.11.0 / 2.11.0 | 0.9-era | same dev build | 1.24.4 | Plausible HLS env; untested. |
| `wsbmr` | 3.10.10 | 2.17 / 3.8 | 0.9.0 | unknown | — | Likely incompatible with TF-2.11-era custom Keras. |
| `fkeras` | 3.12 | scientific stack absent | — | — | — | Name exists, but not usable FKeras. |
| base | 3.12.8 | absent | absent | absent | absent | Not usable. |

`netsurf` also has pandas 2.2.3, SciPy 1.14.0, and tensorflow-model-optimization 0.8.0. `qkeras -> /home/manuelbv/qkeras`; its source is present but not a Git repository, so exact provenance is unknown. It was not modified.

No NVIDIA runtime was available. The small test ran on CPU. Genus 21.1, Innovus 20.1, Calibre 2021.2, Vivado 2023.2, and Vitis HLS 2024.1 are visible; licenses were not tested. Historical reports use a Genus 20.12-era flow, so executable presence is not reproduction.

## Repository state

- About 300 GB: datasets 263 GB, `benchmarks_old` 23 GB, `benchmarks` 8 GB, `syn` 5.4 GB, tutorials 427 MB.
- Dirty handoff: modified configuration, benchmark/dataset/injection/model/plot source, and two notebooks. The key uncommitted model change shrinks TinyML hidden layers from 128 to 10 and adds alpha layers; this is a separate lineage.
- Manuscript, images, qkeras link, and notebook backups are untracked.
- Nothing was reset, cleaned, overwritten, or migrated.

## Current benchmark inventory

All current ranked artifacts use `q<6,2,1>`: six-bit signed two's-complement, range `[-4,3.875]`, step `0.125`.

| Benchmark | Architecture/task | Trainable values / bits | Models | Rankings/results | Hardware | Manuscript identity / reproduction |
|---|---|---:|---|---|---|---|
| `autompg` | FNN 7→64→64→1 regression | 5,121 / 30,726 | `.keras` + weights | 9 methods; 12,051 qPG rows | None | Not Table II; historical software only. |
| `fashion_mnist_fnn` | Flatten 784→64→10 classifier | 51,018 / 306,108 | model + manual HLS tree | 9 methods; 12,055 qPG rows | No corresponding `syn` top | Not Table II. |
| `mnist_fnn` | Flatten 784→64→10 classifier | 51,018 / 306,108 | model | 9 methods; 12,055 qPG rows | No matching top | Table II says 7,845 parameters; identity unresolved. |
| `fashion_mnist_hls4ml` | Small CNN classifier | 11,060 / 66,360; 60,492 nonpruned | model + HLS model | qPG only; 34,603 rows; 684 NaNs, 49,748 zeros | RTL + full-TMR Genus | Artifact quality needs investigation. |
| `keyword_spotting` | DSConv audio classifier | manuscript says 24,193; unverified | model + HLS model | **No current ranking/results** | RTL + full-TMR Genus | Manuscript result provenance missing. |
| `mnist_fnn_lite` | Flatten 784→10→10 | 7,980 / 47,880; 47,070 nonpruned | model + HLS model | qPG; 120,000 rows | RTL + full-TMR Genus | Likely hardware proxy, not traceably same as manuscript MNIST-FNN. |
| `smartpixel_large` | Dense SmartPixel classifier (`dnn/smart_pixel.py:111`) | 3,043 / 18,258; 16,266 nonpruned | model + HLS model | qPG; 30,000 rows | RTL + full-TMR Genus | Strong HEP-adjacent artifacts; no fabrication proof. |
| `tinyml_anomaly_detection` | Current dirty source: 320→10…8…10→320 AE | 8,560 / 51,360; 47,250 nonpruned | model + HLS model | qPG; 45,207 rows | RTL + full-TMR Genus | **Not** manuscript's 179,959-param model. |

`benchmarks_old` contains MNIST/AutoMPG/LeNet-era sweeps (247 ranking and result CSVs), not archived manuscript-scale TinyML or Keyword results.

## HEP provenance

- `dnn/econ.py` contains an ECON-T/HGCAL-style autoencoder and `benchmarks.yaml` has `ECONT_AE`, but no HGCal dataset, checkpoint, result directory, HLS, RTL, or synthesis workspace was found. Code is not proof of an ASIC reproduction.
- SmartPixel has data, model/checkpoint, qPG results, HLS model, RTL, and TSMC65 Genus workspace. This is the strongest reuse chain, but lacks a tapeout/fabrication manifest.
- Released mitigation area cannot be assumed convertible into precision; that needs new retraining/HLS/synthesis under timing, routing, and memory constraints.

---

# D. Scientific consistency audit

## Bit semantics

`core/quantization.py:52-298` implements signed fixed-point two's-complement. For `q<6,2,1>`, array order is sign/MSB through LSB; ranking bit labels are `2,1,0,-1,-2,-3`. Quantization uses NumPy round-to-nearest-even and saturation, not wraparound. The sign delta is `-4` for 0→1 and `+4` for 1→0.

An exhaustive test compared `decode(flip(encode(w),k))-decode(encode(w))` to `compute_delta_matrix()` for all 64 codes × six bits: all 384 matched and all values round-tripped. Manuscript Equation (3), `(-1)^W[k] 2^k`, has the opposite sign for a zero sign bit if `k` includes it. This is a confirmed equation defect. Exact QKeras conformance at half-step ties/learned-alpha layers remains to be tested.

## qPolarGrad as implemented

For dense `w_ij`, bit `k`, the code approximately computes:

```text
P_ijk = mean_x[x_i * delta_ijk]
G_ij  = |dL/dw_ij|
score_ijk = |P_ijk * G_ij|
```

Bias `P` is its delta. Convolution uses a norm of the convolution output from a one-bit delta. Gradients use a cloned model with `training=True`, include regularization losses, and are copied to all bits unless `use_delta_as_weight` is selected (`ranking.py:1099-1190`, `:2139-2185`; dense impact `dnn/layers.py:968-1022`).

This is not elementary `|gradient*delta|`; it adds an input/drift factor. Dense code averages signed drift before absolute value, so input cancellation can zero a score. Units are not a direct loss change. Equation (7) prints `P * grad(L)^T * Delta(theta)` while Table I already puts delta in `P`; read literally, the paper counts delta twice, while code counts it once. The equation/implementation must be re-derived.

The training-state-agnostic claim is unsupported: the score multiplies by the trained-point gradient and collapses near stationary points. Stored rankings contain many zeros.

qPolar also performs a global all-bit attack and computes per-layer impact from globally corrupted activations (`ranking.py:1981-2040`), not clean ones. `_attack()` mutates `*_N` with `assign_add`. Conv impact at `dnn/layers.py:1202-1204` reads persistent `self.kernel_N`, not supplied `N[self.kernel.name]`; repeated calls can scale results. This is a confirmed state/reproducibility defect.

## Ranking behavior and baselines

qPG is not a global susceptibility order. It lexicographically protects all nonpruned sign/MSBs, then the next bit positions, using susceptibility only within a bit tier (`ranking.py:2108-2127`). Pruned values sort last.

Default aliases are reversed (`config/defs.py:54-56`): bit labels rise from `-3` to `2`, but `bitwise_msb` uses `ascending=True`, ordering `-3,-2,-1,0,1,2`. Stored MNIST `bitwise` data labeled `msb_to_lsb` begins at `-3`; qPG begins at `2`. At one tier (51,018 values), stored masks are disjoint, symmetric difference 102,036. The stored baseline cannot support the manuscript MSB/FORTUNE comparison. A correct true-MSB baseline and qPG's position-first policy must have identical masks at exactly one-MSB-per-value coverage.

Implemented methods: random, absolute value, layerwise, bitwise, gradient, gradient×delta, Hessian, Hessian×delta, qPolar, qPG, Fisher, AIBer, Laplacian/spectral (`ranking.py:555-570`). Hessian is a local power-iteration adaptation inspired by FKeras; no exact FKeras provenance was recovered. FORTUNE is not independently implemented; the paper equates it with an MSB point.

## Fault/statistical model

The active injector applies independent Bernoulli flips to unprotected logical trainable-parameter bits. Multiple bit flips are summed. Protection is perfect. It omits activations, compute transients, stuck-at faults, spatial correlation, multi-cell upsets, repair/retry, voter faults, and calibrated cross-sections.

Base weights are not mutated. With `batch_size=None`, one fault mask is shared by the supplied dataset; with batching, one new mask is shared per batch (`injection.py:601-626`). Thus this is neither persistent storage corruption nor per-inference independent BER, and results depend on batch size.

Other confirmed issues:

- no stored run seed or fault-mask ID; no paired realizations across policies;
- unseeded DataFrame randomization breaks tie reproducibility;
- default evaluation uses training data (`experiments.py:755-818`);
- `true_ber` subtracts a constant counter never updated (`:923,1002-1010`), and its denominator conflicts with shared masks/pruned bits;
- online display variance (`:1030-1036`) is not correct Welford/population/sample variance;
- later aggregation provides mean/std but no formal CI;
- protection/BER arrays act as bin edges; default stopping is only two samples/bin, insufficient for “tight confidence intervals.”

BER here is a logical Bernoulli probability for a forward-call window, not an SEU rate, fluence response, FIT rate, cross-section, or time-based reliability quantity.

## Finding register

| ID | Classification | Claim/location | Actual behavior/evidence | Severity | Action |
|---|---|---|---|---|---|
| S1 | Confirmed defect | Eq. (3) | Wrong sign-bit delta vs exhaustive codec test. | Critical | Correct equation; add exhaustive test. |
| S2 | Confirmed defect | Fig. 7 MSB/FORTUNE | `bitwise_msb` protects LSB first; masks disjoint. | Critical | Invalidate/regenerate with mask hashes and paired faults. |
| S3 | Inconsistency | Eq. (7)/Table I | Paper appears to count delta twice; code `|P*grad|`. | Critical | Re-derive and approve estimator. |
| S4 | Unsupported claim | Sec. III-E | Code depends directly on gradient/training point. | High | Remove claim or validate a revised metric. |
| S5 | Confirmed defect | qPolar path | Corrupted activations/persistent `*_N`; Conv uses state rather than argument. | Critical | Reset/invariance/clean-activation tests, then repair. |
| S6 | Confirmed incompatibility | Current software | qPG fails at `ranking.py:1998`. | Critical | Minimal compatibility patch; do not upgrade first. |
| S7 | Confirmed inconsistency | Figs. 5–6 “test” | Default evaluates training data. | Critical | Explicit immutable split; regenerate. |
| S8 | Unsupported interpretation | BER/fault model | Mask is per batch/call, not per inference or physical SEU. | Critical | Specify and implement approved windows. |
| S9 | Confirmed defect | `true_ber` | Counter baseline never updated; denominator ambiguous. | High | Deterministic counter tests/fix. |
| S10 | Unsupported claim | “tight CI” | Mean/std, two samples/bin, unseeded sampling. | High | Define estimand, seeds, pairing, CI/stopping. |
| S11 | Confirmed inconsistency | Table IV/Fig. 4 | Tiny power `R²=0.96194`, MAE 5.36% baseline, endpoint max error 11.36%; not ≥.999/≤.4%. | Critical | Correct and archive raw reports. |
| S12 | Confirmed inconsistency | Figs. 5–6 budgets | Table math yields MNIST 37.8/45.7%, Tiny 86.6/83.5%, not 46/30 and 41/62. | Critical | Single tested cost function/data source. |
| S13 | Unsupported provenance | Table IV | Current reports differ; intermediate reports absent. | High | Recover archive or mark nonreproducible. |
| S14 | Inconsistency | Table II | MNIST/Tiny software/hardware variants do not match table identities. | Critical | Immutable cross-domain benchmark manifests. |
| S15 | Potential improvement | O(N) claim | Conv loops per bit/kernel site; complexity depends on architecture/samples. | Medium | Define N; report measured operations/time/memory. |

---

# E. Synthesis-flow audit

## Recovered flow and guarantees

1. NetSuRF `QModel` is manually converted to ordinary QKeras in notebooks.
2. Learned alpha/beta is intended to fold into preceding weights/biases.
3. A custom hls4ml Catapult backend emits resource-oriented ASIC HLS/RTL with six-bit fixed point.
4. Notebook code manually generates reconfigurable register memories/top interfaces; checked-in RTL is under `syn/inputs/hdl`.
5. Cadence Flowtool/Genus reads HLS RTL plus `reconfigurable_mem`/top wrappers.
6. `triplicate_instance_registers` selects weight, bias, and sometimes alpha/batchnorm scale `mem_reg` hierarchies.
7. `TMR.tcl` generates three sequential cells and one combinational majority voter per output (`TMR.tcl:102-212`).
8. Genus produces pre/post-TMR netlists, DBs, timing, area, and vectorless power reports.

This is replicated storage, not replicated MAC/datapath computation. The voter, common clock/reset/control, downstream computation, correlated faults, and common-mode failures are unprotected. Perfect logical suppression is stronger than hardware TMR.

The target is a dedicated/reconfigurable-register instrumentation architecture, not SRAM. Its TMR costs do not generalize to SRAM/BRAM ECC/parity or memory macros.

## Available evidence

Workspaces exist for Fashion CNN, Keyword, MNIST-lite, SmartPixel, and TinyML-small. TMR reports select 54,765; 138,240; 63,440; 17,100; and 3,965 mapped sequential cells respectively. Counts differ from software nonpruned bits because of synthesis optimization; no machine-readable logical-to-physical manifest exists.

| Design | Current mapped full-TMR area (µm²) | Table IV match? |
|---|---:|---|
| MNIST-lite | 3,426,388 | No; Table 100% is 3,249,424. |
| TinyML-small | 1,825,605 | No; Table 100% is 3,536,847/different model. |
| Keyword | 9,901,217 | Not tabulated. |
| SmartPixel | 1,230,850 | Not tabulated. |
| Fashion CNN | 3,976,046 | Not tabulated; notebook endpoint is close. |

In sampled MNIST reports, `area.summary.preTMR.rpt` and `area.summary.rpt` show the same full-TMR area, so the named pre-TMR report is not a reliable zero point. No 20/40/60/80% report set matching Table IV was found. Preserved MNIST `*.ERROR.db` checkpoints show earlier failed TMR steps followed by later post-TMR outputs; they are provenance warnings.

No completed design DEF/GDS/extracted SPEF or Calibre DRC/LVS output was found. These are synthesis estimates, not post-layout signoff or fabricated silicon.

## HLS conversion risk

`tutorials/7_hls_complete.ipynb` likely transfers weights incorrectly: after creating new QKeras layer `ly`, it reads `ly.trainable_variables` rather than the source NetSuRF layer, potentially folding alpha into new initialization. `tutorials/6_hls.ipynb` records a layer-count checkpoint mismatch. Fixed-vector equivalence is required at NetSuRF, folded QKeras, HLS C, RTL, and netlist stages.

## Cost audit

Recomputed from Table IV:

| Series | OLS R² | OLS MAE / zero cost | Max endpoint relative error |
|---|---:|---:|---:|
| MNIST area | 0.999938 | 0.374% | 0.483% |
| MNIST power | 0.999972 | 0.290% | 0.364% |
| Tiny area | 0.999981 | 0.086% | 0.176% |
| Tiny power | **0.961941** | **5.363%** | **11.360%** |

For overhead budget `B`, endpoint interpolation gives `p=B/((C100-C0)/C0)`:

| Benchmark | 60% area | 80% power | Figure claim |
|---|---:|---:|---|
| MNIST | 37.81% | 45.70% | 46% area / 30% power |
| Tiny | 86.60% | 83.49% | 41% area / 62% power |

Figure values cannot come from Table IV under the stated definition. Tiny 62% approximates a 60% power budget (62.62%), suggesting mixed constants/labels. Notebook plotting uses different endpoints and budgets (Tiny 65% area / 170% power), and one cell labels Fashion constants as SmartPixel.

---

# F. Mitigation implementation plan

Do not implement until storage organization, fault persistence, detection handling, and common cost metrics are approved.

| Method | Reusable infrastructure | Required definition | Smallest valid experiment | Difficulty |
|---|---|---|---|---|
| TMR | Rank masks, register RTL, TMR Tcl, Genus | Storage vs compute; voter reliability/placement; common-mode/control | Small dense layer, exact bit list, deterministic single/double replica faults; synthesize 0/selected/100% | Medium |
| Parity | Packing, injector, evaluator, memory RTL | Word width/layout; error action; scrub/retry/halt; multiplicity | Pack 6-bit weights into chosen word; exhaustive 1/2-bit flips; synthesize checker | Medium |
| SECDED | Packing/fault harness | Actual codeword/shortened code; pipeline; DUE policy; interleaving/scrub | One word: exhaustive single correction/double detection, small network, synthesize codec/storage | Medium-high |
| ABFT | Graph, matrix kernels, evaluator | Supported GEMM/conv; trusted checksum; detection vs correction; retry/recompute | One dense GEMM/checksum under a single compute/storage fault; synthesize checker | High/architecture-specific |
| Ranger | Activation traversal, quantizer, metrics | Sites; calibration split/ranges; source fault; comparator placement | One quantized ReLU tensor with activation and weight flips; synthesize clamp | Medium to medium-high |

Fair comparison needs:

```text
(fault site, persistence/window, multiplicity/correlation,
 protected physical bits/words, detection/correction/recovery,
 latency, area, dynamic/leakage power, timing,
 application metric, confidence interval)
```

Do not count detection-only parity/ABFT as correction without recovery. Pack 6-bit weights into realistic words for ECC/parity; per-weight ECC is unjustified unless it is the actual architecture.

---

# G. Activation extension feasibility

Reusable: `QuantizationScheme`, custom graph/activation lists, gradients, metrics/result storage, and some HLS layer metadata. Missing: activation delta/rank/injection state, physical allocation, lifetime, occupancy, and reuse.

`|dL/da * delta_a|` is a reasonable local estimator but is input-dependent and not directly comparable to persistent weight benefit. Fair physical ranking additionally needs site-specific quantizers; held-out activation distributions; buffer/register banking and word widths; residence time/vulnerable cycles; physical upset exposure; fan-out and temporal propagation; overwrite/flush behavior; per-site protection cost; and a common benefit/cost objective.

Implement activation injection first as a separate domain. Do not create a unified ordering until exposure/cost are approved.

---

# H. HEP application feasibility

## ECON/HGCAL

**Low viability without external recovery.** Model source exists, but no dataset, checkpoint, HLS, RTL, synthesis reports, or exact tapeout/publication manifest. Applying NetSuRF now would create a new model, not reproduce a historical ASIC. Needed externally: exact checkpoint/model hash, preprocessing/dataset version, quantization, HLS/synthesis commit/constraints, RTL, technology manifest, radiation environment/cross-sections, and detector metric.

## SmartPixel

**Moderate viability for a new case study; low for a fabricated-ASIC reproduction claim.** Data, model/checkpoint, qPG results, HLS model, RTL, TMR mapping, and TSMC65 Genus reports exist. This supports a controlled reproduction attempt after stabilization, not a fabrication claim.

A defensible HEP study needs a frozen detector task/model; radiation-calibrated assumptions; detector-level performance; verified physical mapping; TMR versus architecture-appropriate ECC/parity; and actual precision/protection retraining and synthesis. “Freed area buys precision” must be tested, not inferred.

---

# I. Refactoring roadmap

## Mandatory stabilization

1. Hash/inventory checkpoints, rankings, results, notebooks, RTL, reports, and dirty diff.
2. Create per-run manifests: source/diff, environment, external commits, dataset/checkpoint, split, quantization, pruning, ranker, seeds, fault model/window, outputs.
3. Add side-effect-free import/inspection; tests must not create DBs/directories.
4. Patch only compatibility blockers: qPG metric handling and checkpoint/path lookup.
5. Add exhaustive codec and exact mask/cardinality/MSB/pruning tests.
6. Make separate stored seeds for initialization, ties, LHS, and fault masks; support pairing.
7. Fix split, counters, variance/CI, stopping, and append correctness.
8. Give MNIST-64/lite and Tiny-128/10 distinct immutable IDs.
9. Validate NetSuRF→QKeras→HLS C→RTL→netlist fixed-vector equivalence.
10. Recover Table-IV raw runs or withdraw them; parse costs from hashed reports.

## Incremental architecture afterward

- `BenchmarkManifest`: immutable dataset/model/checkpoint/split lineage.
- `BitCodec`: encode/decode/flip/saturation/quantizer conformance.
- `FaultModel`: site, distribution, persistence, correlation, seed.
- `InjectionTarget`: weight, activation, compute output, physical word.
- `SusceptibilityEstimator`: scores only.
- `ProtectionPolicy`: scores + costs/constraints → mask.
- `Mitigation`: TMR/parity/ECC/ABFT/Ranger semantics/recovery.
- `HardwareBackend`: HLS/RTL/Genus/FPGA.
- `ExperimentPlan`: pairing/stopping/CI.
- `ResultStore`: versioned append-only schema + hashes.
- `ReportBuilder`: reads, never mutates results.

Separate estimator from the hard-coded MSB-tier policy so global qPG, MSB-first qPG, and cost-aware policies can be compared.

Use incremental stabilization, not a clean-slate rewrite. Preserve custom layers/hardware knowledge. After one benchmark reproduces, extract interfaces around old code. A larger redesign is justified for statistics/orchestration and hardware manifests, where current coupling prevents fair comparison.

---

# J. First executable milestone

## Definition

Use `mnist_fnn_lite` only after freezing/renaming its identity:

1. load one exact checkpoint without training;
2. evaluate a fixed hashed test subset;
3. generate qPG twice with identical seeds and compare score/rank hashes;
4. materialize 0%, one true-MSB tier, and one partial mask; record bit tuples/hashes;
5. run paired deterministic masks for qPG and correctly configured MSB;
6. prove weights unchanged and per-trial counts correct;
7. parse verified zero/full synthesis reports and link selected physical registers to software variables;
8. emit one immutable manifest plus compact JSON/CSV.

## Test executed

A disposable test used Python 3.10.16 / TF 2.11.1, `q<6,2,1>`, a 2→2→1 FNN, four examples, fixed Python/NumPy/TF seeds, and `/tmp` for HOME/DB/config/ranking/logs. Runtime was seconds on CPU.

- Import/model construction: passed, while confirming import side effects.
- Exhaustive codec/delta test: passed.
- qPG: failed reproducibly at `ranking.py:1998`.
- Mask/injection repeatability: not reached.

No-metric failure: empty floating index array. Normal `mse` failure: retained metric string where Keras expects `.name`; recompilation with a metric object did not replace the custom string. Current source/environment is incompatible.

## Proposed commands after minimal patch

First commit a tiny regression suite and run:

```bash
cd /home/manuelbv/netsurf
env CUDA_VISIBLE_DEVICES=-1 TF_DETERMINISTIC_OPS=1 \
  conda run -n netsurf python -m pytest -q \
  tests/test_quantization_q621.py \
  tests/test_qpolargrad_smoke.py \
  tests/test_ranking_masks.py \
  tests/test_injection_reproducibility.py
```

Then a bounded checkpoint run in a new root, never historical `benchmarks/`:

```bash
cd /home/manuelbv/netsurf
env CUDA_VISIBLE_DEVICES=-1 TF_DETERMINISTIC_OPS=1 \
  conda run -n netsurf python -m netsurf \
  --benchmark mnist_fnn_lite \
  --quantization 'q<6,2,1>' \
  --prune 0.125 \
  --benchmarks_dir /home/manuelbv/netsurf/audit_runs/milestone_001 \
  --datasets_dir /home/manuelbv/netsurf/datasets \
  --method qpolargrad \
  --method_suffix audit_milestone \
  --method_kws ascending=False absolute_value=True batch_size=96 \
  --num_reps 3 \
  --protection_range 0.0 0.1666666667 \
  --ber_range 0.001 \
  --injection_batch_size 1000 \
  --no-save_benchmark \
  --no-save_experiment
```

Do **not** run that CLI yet: add explicit `--seed`, `--dataset_split test`, paired-fault support, maximum iterations, and exact checkpoint selection. Today the ranges are bin edges and the loop may run far beyond three reps.

For hardware, initially parse existing reports only. Require verified zero/full reports, TMR list, tool/constraint manifest, and hashes. The named pre-TMR report is inadequate because it repeats full-TMR area.

---

# Required conclusions

## Five highest-priority findings

1. The MSB/FORTUNE comparison is invalidated by reversed `bitwise_msb` configuration and demonstrably different stored masks.
2. qPG is not executable in the intended environment, and its expression/state flow differs materially from Equation (7).
3. The experiment is a batch-window logical Bernoulli model evaluated on training data by default, not a per-inference calibrated SEU/test model.
4. Table IV, Figures 5–6, and checked-in synthesis evidence conflict; Tiny power refutes the stated fit quality.
5. Provenance is broken across software/hardware variants, especially MNIST and TinyML.

## Immediate engineering milestone

Complete the deterministic bounded MNIST-lite milestone after a minimal compatibility patch, exact checkpoint/split manifest, correct MSB baseline, and fixed counting/statistics. Do not add mitigation methods first.

## Minimum corrected preprint work

- resolve Equations (3)/(7) and qPG clean/state semantics;
- rerun qPG, true-MSB, Hessian, and random with exact paired masks/faults;
- use held-out test data and report seeds, CIs, sample counts, and windows;
- manifest all benchmark variants;
- correct/withdraw Table IV and Figures 4–6 from traceable evidence;
- narrow claims to logical weight-bit susceptibility and register-storage TMR;
- add empirical weight-vs-activation tests under a stated exposure model;
- publish environment lock and regression suite.

## Additional TODAES-level work

- realistic accelerator/storage architecture and calibrated fault model;
- actual synthesized TMR, packed parity/SECDED, architecture-valid ABFT, and Ranger cost/latency/power;
- activation exposure and transformer benchmark under common definitions;
- logical-to-physical validation and intermediate hardware points;
- correlated/multi-bit, voter/common-mode, and DUE handling;
- powered paired experiments on challenging tasks;
- independent table/figure regeneration from immutable artifacts.

## HEP path viability

SmartPixel is conditionally viable and better supported than ECON/HGCAL. It could support a new JINST/TNS/NIM-A study after provenance, radiation assumptions, detector metrics, software/HLS equivalence, and mitigation costs. ECON/HGCAL needs external checkpoint/data/hardware recovery. Neither supports a prior-fabricated-ASIC reproduction claim today.

## Decisions requiring scientific approval

1. qPG definition: code `|P*grad|`, elementary `|grad*delta|`, corrected Eq. (7), or another estimator.
2. Position-first versus global score/cost protection policy.
3. Target hardware: dedicated registers, SRAM/BRAM words, or named memory hierarchy.
4. Fault domains/windows: persistent weights, per-inference/batch transients, activations, compute, correlated and voter/common-mode faults.
5. Physical exposure/radiation normalization.
6. Detection recovery semantics for parity/ECC/ABFT.
7. Common cost objective and hard latency/timing constraints.
8. Canonical benchmark/checkpoint/split set.
9. General CAD versus single HEP application direction.
10. Withdraw, label archival, or regenerate unprovenanced historical numbers.

Stop after these approvals and the first deterministic milestone; only then begin mitigation implementation or restructuring.
