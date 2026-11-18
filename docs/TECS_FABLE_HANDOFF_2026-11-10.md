# NetSuRF Paper 1: Fable takeover map (10 November 2026)

This is a navigation and reproducibility handoff, **not** a scientific result manifest or approval to run new campaigns. Fable's review is at `<repo>/docs/TECS_INTERNAL_REVIEW_2026-11-10.md`. It had only the two PDFs, not the code or saved fields. Check each criticism against the artifacts before changing a numerical claim. Its recommendation was **major revision**, with a particularly serious concern that the featured ranking reversals may occur where task performance is already unusable. The review's suggested experiments are proposals, not completed evidence.

## Authority, ownership, and immediate safety

Read `<repo>/AGENTS.md`, `<repo>/docs/SCIENTIFIC_CONTRACTS.md`, the review above, and this worktree's `publications/paper1_review/EVIDENCE_HANDOFF.md` first. Mathematical definitions and verified executable behavior precede finalized hashed evidence, scientific contracts, historical intent, and drafts. Logical BER is **not** a physical radiation rate; ideal event suppression is **not** implemented ECC/TMR/ABFT. Do not modify historical bundles, asset locks, or frozen plans. Do not pool different cohorts, precisions, backends, fault schedules, or task metrics.

Canonical repository: `<repo>`, branch `v2` (observed HEAD `e3209f7f67ef8f30cebb0eed791bf817164d39c0`). It has researcher-owned untracked files; leave it alone. Current manuscript worktree: `<repo>/workspace/implementation-worktrees/paper1-complete-review-v1`, branch `paper1-complete-review-v1`, observed HEAD `3c517dc54d8f3dc48dfd57d4e94fb0a7fa315b09`. Revise from this source in a **new isolated worktree/branch**; do not overwrite its current delivery, merge, or push automatically. Use the researcher's configured Git identity. All new analyses and outputs should be versioned/write-once under that new worktree's gitignored `workspace/`. Before scheduling a large job, inspect live processes and provide a bounded resource and scientific plan to the researcher.

## Current paper, original paper, and delivery

The authoritative current LaTeX source is `<repo>/workspace/implementation-worktrees/paper1-complete-review-v1/publications/paper1_review/source/main.tex`. Its numbered sections are under `source/sections/`, preamble and metadata under `source/include/`, diagrams under `source/figures/`, bibliography at `source/references.bib`, and numerical figure PDFs under the sibling `publications/paper1_review/figures/`. `source/README.md` explains Overleaf and local building. The current title is *Is There a Universal Bit Ranking? Evaluating Fault Protection in Quantized Neural Networks*; Fable questions whether a finite comparison supports that framing. The manuscript is a 14-page ACM internal-review layout, not a venue-approved final submission.

Current frozen delivery directory: `<repo>/workspace/implementation-worktrees/paper1-complete-review-v1/workspace/paper1_review/delivery_v4/`. It contains `NetSuRF_internal_review.pdf`, `NetSuRF_Overleaf_source.zip`, `NetSuRF_complete_review.zip`, `NetSuRF_full_grid_supplement.pdf`, `delivery_manifest.json`, and `checksums.json`. The internal PDF SHA-256 is `a929b55914bcd04a4b4b5f44d07258acd45a18d621c857b2309999466aeb9a65`; the complete-review ZIP SHA-256 is `42d5e7f99541f84b67e30172cfcd4bcb45c43f001ac69aa3423c92c9e5ed85c6`. Preserve `delivery_v4`; use a new delivery destination.

The rejected TCAD R1 proof supplied by the researcher is `<home>/.codex/attachments/4b72e0b5-18ef-40c8-8b51-4a911f026e94/TCAD-2026-0117.R1_Proof_hi.pdf`. The historical source ZIP is `<home>/.codex/attachments/765216ef-6301-4016-b790-a5a77b118913/ACM_TECS___NetSuRF__Encoding_Aware_Fault_Susceptibility_and_Budgeted_Protection_for_Quantized_Neural_Networks.zip`. These document intent and tone, **not validated v2 results or hardware claims**. The current manuscript is software-only, persistent stored INT8 weight-bit faults, and ideal logical protection. Do not resurrect old qPolarGrad, delta-matrix, partial-TMR area/power, or radiation claims absent valid current evidence. The original-to-current audits are in `publications/paper1_review/review/methods_and_original_audit.md` and `legacy_figure_reuse.md`.

Safe local build, using existing TeX Live/acmart:

```bash
cd <repo>/workspace/implementation-worktrees/paper1-complete-review-v1/publications/paper1_review/source
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The current complete package was built with `publications/paper1_review/scripts/package_review.py`; it accepts `--out NEW_WRITE_ONCE_DIR`, `--original-zip`, and `--original-pdf` and refuses an existing destination. Consult `--help` and the script before using it. Building in a fresh worktree avoids changing tracked generated figure files.

## Six main studies and what the numbers mean

The main source locator is **`publications/paper1_review/data/figure_provenance.json`** in the manuscript worktree. It records absolute source paths and SHA-256 for figure inputs, manifests, field-level trials, configs, and selected analyses. Start there, then verify each source's `manifest.json`, checksums/finalization marker, `scientific_payload.json`, frozen config/plan, and saved policy-field observations. The full study description and computation costs are in `EVIDENCE_HANDOFF.md`.

| Study | Eligible stored INT8 bits | Evaluation cohort | BER values | Fields per BER | Methods | Measured conditions | Labeled policy-field rows |
|---|---:|---|---|---:|---:|---:|---:|
| ResNet-8 / CIFAR-10, TFLite | 618,880 | 1,000 images | 8 | 32 | 7 | 240 | 7,680 |
| DS-CNN / Speech Commands, TFLite | 176,128 | 1,000 clips | 8 | 32 | 7 | 240 | 7,680 |
| ToyCar autoencoder, TFLite | 2,113,536 | 64 complete files | 5 | 32 | 7 | 150 | 4,800 |
| DistilBERT / SST-2, W8A8 | 344,469,504 | 512 examples | 8 | 24 | 6 | 208 | 4,992 |
| Pythia-160M / WikiText-2, W8A8 | 988,545,024 | 4,096 scored tokens | 8 | 24 | 6 | 208 | 4,992 |
| Qwen3-0.6B / WikiText-2, W8A8 | 4,767,875,072 | 4,096 scored tokens | 8 | 24 | 6 | 208 | 4,992 |

All use 6.25%, 12.5%, 18.75%, 25% interior coverage plus zero/full controls. Except ToyCar, BERs are `1e-5, 1e-4, 3e-4, .001, .003, .01, .03, .1`; ToyCar uses `1e-5, 1e-4, .001, .003, .01`. These are **six separate frozen studies**, 1,254 condition rows and 35,136 labeled policy-field records; repeated policy use of one proposed field is paired, not a new independent field. ResNet/KWS/ToyCar include published-form FKeras; the W8A8 studies do not. PrioriFI-derived allocation has no finalized matched effectiveness study in this main grid. FKeras and MSB produce identical masks at complete INT8 significance tiers (12.5% and 25%); do not count those as independent treatment decisions.

Classification reports cross-entropy and accuracy separately. ToyCar's authoritative risk is complete-file `1 - ROC-AUC`, *not* reconstruction MSE; MSE is a gradient surrogate. The language-model loss is token NLL with a 30-nat penalty for nonfinite failures; failed-token fractions, cap frequencies and finite-token NLL must remain visible. All main intervals condition on a frozen checkpoint, evaluation cohort, and constructed masks. The TFLite random baseline has four mask/schedule replications; the deterministic rankings have one mask per budget. The model families and task metrics must never be averaged as though they share a unit. BER and expected applied fault count should both be reported.

Finalized primary source manifests, all cross-indexed in `data/figure_provenance.json`:

1. ResNet: `<repo>/workspace/implementation-worktrees/m12-fkeras-multibench/workspace/runs/m12_resnet8_fkeras_matched_grid_v1/20261105T175030.135296Z-c799a357/manifest.json`
2. KWS: `<repo>/workspace/implementation-worktrees/reliability-db/workspace/runs/kws_partial_tier_fkeras_matched_v1/20261104T042124.586895Z-b0284b42/manifest.json`
3. ToyCar: `<repo>/workspace/implementation-worktrees/m12-fkeras-multibench/workspace/runs/m12_toycar_fkeras_matched_grid_v1/20261105T210814.691588Z-16296f36/manifest.json`
4. DistilBERT: `<repo>/workspace/implementation-worktrees/w8a8-ber-stress/workspace/runs/w8a8_wide_ber_matched_v1/distilbert_sst2_int8/manifest.json`
5. Pythia: `<repo>/workspace/implementation-worktrees/pythia-wide-confirm/workspace/runs/pythia_w8a8_high_ber_clean_gated_v2/manifest.json`
6. Qwen: `<repo>/workspace/implementation-worktrees/w8a8-ber-stress/workspace/runs/w8a8_wide_ber_matched_v1/qwen3_0_6b_int8/manifest.json`

The separate exploratory Qwen crossover is `<repo>/workspace/implementation-worktrees/w8a8-followup/workspace/runs/qwen_crossover_p05_p075_v1/manifest.json`; it is **not** another main-grid replication. Additional focused FKeras and conditional studies are located by the same provenance JSON and by `review/figure_table_manifest.md`.

## Databases, analysis data, and scripts

The latest curated **SQLite v3 is older than the manuscript-wide grid**: `<repo>/workspace/implementation-worktrees/reliability-db/workspace/m12_current_comparative_v3/current_comparative.sqlite`. Its `primary_conditions` view has 280 historical rows; `current_comparative_conditions_v2` has 552; `current_comparative_conditions_v3` has 864 (630 current TFLite + 234 earlier low-BER W8A8). It does **not** contain all 1,254 manuscript wide-BER conditions. Do not silently use it as the source for Fig. 2 or 3, or mutate this finalized database. Its `README.md`, `provenance.json`, `checksums.json`, `study_registry.csv`, `current_conditions.csv`, `current_trials.csv`, `paired_contrasts.csv`, `coverage_matrix.csv`, and `method_coverage.csv` explain exactly what it contains. Database implementation: `<repo>/workspace/implementation-worktrees/reliability-db/netsurf/reliability_database_v3.py` and `workflows/build_reliability_database_v3.py`. A newer comprehensive SQLite, if needed, must be a **separate versioned derived product** assembled from verified raw fields.

Read-only SQL example:

```bash
sqlite3 -readonly <repo>/workspace/implementation-worktrees/reliability-db/workspace/m12_current_comparative_v3/current_comparative.sqlite \
  'SELECT study_id, COUNT(*) FROM current_comparative_conditions_v3 GROUP BY study_id ORDER BY study_id;'
```

The current manuscript's combined offline data are under `publications/paper1_review/data/`:

- `reliability_conditions.csv`: 1,254 condition-level means/intervals; inspect `model,method,coverage,ber,mean,low,high,clean,fields,source`. Plot source, not a substitute for raw-field verification.
- `wide_paired_comparisons.csv`, `fkeras_paired_comparisons.csv`: saved paired contrasts; preserve the prespecified comparison families and Holm conventions.
- `empirical_validation.csv`, `empirical_partitions.json`, `empirical_provenance.json`, `toycar_auc_correlations.csv`: 320 shared stratified exact-bit candidates/model and sample-limited score audits; not uniform whole-network estimates.
- `kws_pool_statistics.json`, `conditional_historical_summary.csv`, `conditional_historical_provenance.json`: bounded conditional selection and interaction analyses, separate from main reliability grids. The 32-pool KWS conditional-minus-exact-isolated held-out mean is `-0.0000824` nats/clip with CI `[-0.0016935, 0.0015287]`; it does **not** establish improvement.
- `qwen_crossover.json`: separate exploratory crossover; do not fold it into the frozen main study.
- `fkeras_numerical_receipts.csv`, `score_construction_costs.csv`, `estimator_runtime.csv`, `fkeras_candidate_audit_v1.json`: implementation and ranking-cost evidence. The candidate-correlation audit has a ResNet result; KWS/ToyCar joins remain blocked by a historical input-hash mismatch, despite valid protection-effectiveness runs.
- `benchmark_inventory.json`, `model_table.csv`, `m11_bounded_scaling_*`: identities and bounded mathematical/engineering evidence; M11 is not a whole-model effectiveness campaign.

Scripts in `publications/paper1_review/scripts/` include `build_evidence_figures.py`, `build_compact_figures.py`, `build_empirical_audit.py`, `audit_fkeras_candidates.py`, `verify_conditional_history.py`, `verify_m11_history.py`, `summarize_inventory.py`, and `package_review.py`. Inspect their arguments and write behavior before invocation: figure generators write named outputs and must **not** be pointed at the frozen `delivery_v4`; `audit_fkeras_candidates.py --check` is the read-only verification route, whereas its normal output uses write-once creation. `review/claim_to_evidence.md`, `review/figure_table_manifest.md`, `review/unresolved_experiment_ledger.md`, `review/bibliography_audit.md`, and `review/fourth_revision_notes.md` are manuscript audit aids, not new scientific authority.

Current main plots (PDF/SVG/PNG under `publications/paper1_review/figures/`): `reliability_ber_125`, `reliability_ber_250`, `reliability_budget_001`, `paired_reversals_and_kws`; `reliability_all_budgets_supplement.pdf` has all 24 budget panels. They show observed cells only; no interpolated FKeras/PrioriFI values. See `data/figure_provenance.json` for exact input hashes. Recompute new reviewer-requested analyses from **saved raw field rows**, not from these condition means alone.

## Code and toolchains

`proj` is `/opt/proj/bin/proj`; repository development docs use `proj activate netsurf` and `python -m pytest -q`. The verified offline analysis interpreter is `<netsurf-env>/bin/python` (Python 3.11.17; Matplotlib 3.11.2). Use this for SQLite/CSV/plots and manuscript analysis. The pinned TFLite inference interpreter is `<assets>/.venv-mlperf/bin/python` (Python 3.11.18; TensorFlow 2.14.1, `BUILTIN_REF`). The pinned transformer W8A8 interpreter is `<assets>/envs/torch-2.9.1-cpu-py311/bin/python` (Python 3.11.18; PyTorch 2.9.1+cpu). **Do not** use the TFLite interpreter for Matplotlib/SQLite reporting, and do not infer that cross-backend models are numerically identical. No package installation is needed for this handoff.

Relevant source-contract docs: `<repo>/docs/ARCHITECTURE.md`, `docs/REPRODUCIBILITY.md`, `docs/MLPERF_TINY_M71_CONFORMANCE.md`, `docs/M97_FKERAS_REPRODUCTION.md`; implementation-branch docs `<repo>/workspace/implementation-worktrees/m12-fkeras-multibench/docs/M12_FKERAS_MULTIBENCH.md` and `M12_FKERAS_MATCHED_PLANS.md`. Core code includes `netsurf/experiment_engine.py`, `netsurf/faults.py`, `netsurf/tflite_storage.py`, and `netsurf/conditional_risk.py`; the actual wide-BER transformer path is `<repo>/workspace/implementation-worktrees/w8a8-ber-stress/netsurf/w8a8_wide_ber_campaign.py`. Check the source revision recorded in each manifest before claiming that canonical `v2` code reproduces it.

Existing production runner examples, **for inspection only; never rerun a finalized study**: `<repo>/workspace/implementation-worktrees/m12-fkeras-multibench/scripts/run_m12_fkeras_study_once.sh` and `<repo>/workspace/implementation-worktrees/w8a8-ber-stress/scripts/run_w8a8_wide_ber_model.sh`. Their frozen configs are in their respective `config/experiments/` directories. The latter separates freeze, shards, and finalization. For new experiments, create a distinct plan/run ID, freeze cohort/model/mask/fault-law identities and source hash, estimate requests/runtime/disk, obtain researcher approval, and use foreground tmux execution only after resource admission. Never reuse untouched final data for calibration.

## Review-driven triage, not a prejudged conclusion

The review's most consequential **offline** checks are: (1) define/display a task-usable operating region and report accuracy/AUC, token failure and cap rates alongside loss; (2) calculate all-cell paired regret and win/loss/unresolved patterns, including the possibility that one gradient ranking is a robust default; (3) audit the sign and arithmetic of the KWS three-mask contrasts—the two reported improvements do not obviously subtract to the reported primary contrast; (4) show mask anatomy and exact protected cardinality, and test whether partial-tier MSB tie-breaking drives apparent FKeras reversals; (5) state the actual Holm families, resampling resolution and uncertainty unit; (6) verify model/checkpoint/eligible-tensor metadata for the appendix; (7) verify TFLite cross-entropy's output and floor semantics from code. The review's attributions to nonfunctional regimes and “harmful protection” are hypotheses to test against saved accuracy/field records, **not** pre-accepted facts.

The conditional KWS experiment changed 27/32 bounded masks without a resolved held-out risk gain; do not sell it as optimizer superiority. Its small effect ceiling and field/pool uncertainty need to be shown if retained. Fable's proposed new independent-checkpoint, within-tier randomization, structured-mask, and conditional-value experiments require separate frozen protocols and resource estimates. Do not automatically run them. Prioritize analyses that can be done **without inference** before deciding whether any new run is needed for the paper deadline. The review did not fully verify the external literature or journal page limit; check original primary sources and current venue instructions before making novelty or format claims.

## Fast, safe orientation commands

```bash
cd <repo>/workspace/implementation-worktrees/paper1-complete-review-v1
git status --short
rg --files publications/paper1_review | rg '(source/main.tex|data/|review/|scripts/)' | head -80
sha256sum workspace/paper1_review/delivery_v4/NetSuRF_internal_review.pdf
<netsurf-env>/bin/python -c 'import csv; p="publications/paper1_review/data/reliability_conditions.csv"; print(sum(1 for _ in csv.DictReader(open(p, newline=""))))'
```

For experiment status, use read-only `tmux list-sessions`, `ps`, and source manifests. A past tmux title or queued progress percentage does not prove that a campaign is finalized. Report any missing source, hash mismatch, unresolved conformance issue, or data-exposure problem explicitly rather than substituting a nearby run.
