# Handoff from NetSuRF to Kona Assets: what NetSuRF uses, and what to add (13 November 2026)

Goal (from the researcher): one shared, checksummed registry for every model, dataset and benchmark that NetSuRF
uses, so other projects can use the same assets. NetSuRF keeps its own scientific code (fault injection, bit
rankings, protection search, gradient companions); Kona Assets supplies frozen, verified assets and locks.

Rules on the NetSuRF side that matter here: NetSuRF run outputs under
`<repo>/workspace/implementation-worktrees/*/workspace/runs/` are write-once evidence; copy from
them, never move or modify them. Historical folders under `<repo>/DEPRECATED/` (including
`external_links/qkeras`) and `/archive/WSBMR/` are read-only evidence too. Gated or restricted material is not copied
(see Section 4).

## 1. Already in Kona Assets and used by NetSuRF (no action, except item 1d)

| Asset | How NetSuRF uses it | Kona Assets status |
|---|---|---|
| MLPerf Tiny ResNet-8 INT8 (IC) | `<assets>/locks/mlperf-tiny/ic.lock.json`; model blob in `/data/models/models--mlcommons--tiny/`; CIFAR-10 at `/data/datasets/cifar10/` | READY |
| MLPerf Tiny DS-CNN INT8 (KWS) | `kws.lock.json`; Speech Commands 0.0.3 at `/data/datasets/speech_commands/0.0.3` (train, validation and test shards are all used: train for calibration, validation for predictions, test for evaluation) | READY |
| MLPerf Tiny ToyCar INT8 (AD) | `ad.lock.json`; `/data/datasets/dcase2020-toycar`, `/data/datasets/toyadmos` | READY |
| DistilBERT/SST-2, Pythia-160M, Qwen3-0.6B-Base W8A8 executables | built by Kona Assets (commit `1aea52b`), stored as `/data/models/local--kona-assets--*`; datasets `/data/datasets/glue_sst2`, `/data/datasets/wikitext2` | READY in `NETSURF_ASSET_READINESS.json` |

1d. The language-model executables have readiness records but no lock files under `locks/`. NetSuRF would like
locks for them in the same `kona-assets.lock` schema (model files, quantization rules, tokenizer, evaluation cohort
and its index, reference metrics), so they are verified the same way as the MLPerf Tiny models.

## 2. Used by NetSuRF now but not in Kona Assets: eight derived ResNet-8 checkpoints

Trained by NetSuRF with the MLPerf Tiny reference recipe and converted to INT8 TFLite; all nine ResNet-8 checkpoints
(pinned plus these eight) carry the paper's main confirmatory results. Paths are relative to
`<repo>/workspace/implementation-worktrees/paper1-tecs-revision-v1/workspace/runs/`:

| Checkpoint | Folder | Recipe script |
|---|---|---|
| Retrained, seeds 7101-7105 | `paper1_runD_resnet8_retrain_v1/seed_710{1..5}/` | `workflows/retrain_resnet8.py` |
| Pruned 50% and 70% of each layer's kernels | `paper1_runJ_resnet8_variants_v1/pruned_s0.5/`, `pruned_s0.7/` | `workflows/train_resnet8_variants.py` |
| Under-trained (100 instead of 500 epochs) | `paper1_runJ_resnet8_variants_v1/undertrained_e100/` | `workflows/train_resnet8_variants.py` |

Each folder holds `trained.h5` (Keras), `trained_quant.tflite` (INT8, the file NetSuRF evaluates), `training_log.csv`
and `manifest.json` (seed, recipe, hashes). The scripts are in the paper worktree above. NetSuRF currently passes
each file's sha256 through the environment variable `NETSURF_RETRAINED_IC_SHA256` to the IC adapter. Wanted: register
them as derived models of the IC benchmark with lineage (parent recipe, seed, pruning or epoch change, training
script and its hash, calibration set for conversion), and a lock per checkpoint or one family lock.

## 3. MLPerf Tiny gaps

- **Visual Wake Words (MobileNetV1-0.25 INT8):** `vww.lock.json` exists but its result is provisional (image
  provenance unverified); NetSuRF's registry marks the candidate `blocked`. Wanted: whatever closes the provenance
  gap, so VWW can join the static and switch tests. Data: `/data/datasets/visualwakewords`, and a legacy
  `vw_coco2014_96` copy under `/data/datasets/netsurf-legacy/`.
- **Streaming Wake Word (1-D DS-CNN):** deferred in Kona Assets; NetSuRF has a partial adapter
  (`netsurf/adapters/mlperf_streaming.py`). Lower priority.

## 4. Legacy NetSuRF benchmarks to bring in (detector ASIC and FPGA models)

The old NetSuRF (QKeras-based) trained and stored these; the researcher wants them registered, ECON-T first, because
it is a real detector-ASIC model and may justify its own paper.

| Model | Where | Notes |
|---|---|---|
| ECON-T autoencoder (CMS HGCAL) | `/archive/WSBMR/common/benchmarks/ECONT_AE/6bits_0int/` (`pruned_…` and `non_pruned_4977_conv8f3k2s_flatten_dense16u_dense128u_reshape_conv8f3k2s_conv1f3k1s_sigmoid/`, each with `models/`, `sessions/`, `experiments/`); also `ECONT_AE_old/` | 6-bit QKeras weights, 4,977 parameters. Training data: `/archive/WSBMR/alan/hgcal22data_signal_driven_ttbar_v11.tar.gz`. A related project exists at `<other-project>`, which links into a gated folder (`/projects/.gated/fnal/tsmc65/…`): respect its access group and copy nothing from it without its owners' approval. |
| SmartPixel models | dataset `/data/datasets/SmartPixel/ds8_only` (group-restricted permissions) and `/data/datasets/netsurf-legacy/repository-datasets-2026-10-22/SmartPixel`; legacy loader `DEPRECATED/legacy_source/netsurf/core/datasets.py` class `SmartPixel` | Benchmark names in the legacy code: `smartpixel`, `smartpixeldense`, `smartpixeldenselarge`. Trained checkpoints not yet located; check `/archive/WSBMR/` and the researcher's earlier in-pixel work. |
| hls4ml MNIST MLP, hls4ml CIFAR-10 | `/archive/WSBMR/common/benchmarks/mnist_hls4ml/`, `cifar10_hls4ml/` | QKeras. |
| hls4ml jet tagging | data `/data/datasets/JetTagging/` (`hls4ml_LHCjet_30p_train/val`) | Model not yet located in the archive. |
| Others in the legacy archive | `/archive/WSBMR/common/benchmarks/`: `keyword_spotting`, `tinyml_anomaly_detection`, `tinyml_person_detection`, `mnist_lenet5`, `mnist_squeezenet`, `cifar10_resnet18`, `autompg` | Lower priority; the MLPerf Tiny ones are superseded by the official INT8 references. |

The legacy archive is about 22 GB (`/archive/WSBMR/common/benchmarks`); the legacy dataset copy under
`/data/datasets/netsurf-legacy/repository-datasets-2026-10-22/` is about 263 GB, mostly `toyadmos`.

For each legacy model NetSuRF needs:
1. the trained checkpoint and training data identified by sha256, with the training configuration;
2. a frozen integer executable with exact stored weight codes and scales: either a TFLite INT8 export or a recorded
   integer forward pass, verified against the QKeras model on a fixed sample (output agreement reported, as for the
   MLPerf Tiny locks);
3. the quantization of every weight tensor (bit width, integer bits, per-channel or per-tensor scale), because faults
   are applied to stored codes;
4. fixed, disjoint sample partitions (calibration, validation, evaluation) with index files and hashes;
5. the task metric and a reference result.

NetSuRF will write its own adapters and gradient companions against these locks.

## 5. Priority

1. Locks for the eight derived ResNet-8 checkpoints (Section 2): they back published results now.
2. ECON-T with its HGCAL data (Section 4).
3. Locks for the three language-model executables (item 1d).
4. VWW provenance (Section 3).
5. SmartPixel, then the hls4ml models.

Contact: the NetSuRF paper session (ACM TECS manuscript review). A separate read-only assessment of the legacy
ECON-T, SmartPixel and MNIST models has been proposed on the NetSuRF side; its report, if run, will be at
`publications/paper1_review/review/LEGACY_ASIC_BENCHMARKS_ASSESSMENT_2026-11-13.md` in the paper worktree.
