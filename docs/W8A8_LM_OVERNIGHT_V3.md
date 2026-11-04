# Whole eligible W8A8 LM campaigns: versioned M12 co-execution

This record covers the Pythia and Qwen W8A8 campaigns launched or prepared on
2026-11-04. The v1 and v2 frozen plans are retained. The v3 plan of each model
changes only the operational resource gate and adds hashes for the measured
wrapper and phase entry. Its `equivalence_audit.json` compares every scientific
field, original source binding, READY snapshot, cohort, fault seed, method and
budget with v1. The immutable source checkpoint, W8A8 conversion, integer
execution and task-risk implementation are unchanged. The v3 plans remain
write-once and use distinct v3 run directories.

| Model | v3 `plan.json` SHA-256 | v3 internal `plan_sha256` | v3 audit SHA-256 |
|---|---|---|---|
| Pythia-160M | `7e8457045cedbc42fecf2435a2f6dc17f9bf11d802348b161cb9bc70faccf54c` | `c3ad6485b3201b7ff5f355860373a56a5d2166d118a0c9c596ca0882af65e843` | `133867888e0db2a3e8dfbe3cca95c23344f43061a30f4f3586dc8416a6f07565` |
| Qwen3-0.6B-Base | `2003d9483a068b55f94d2f2f46d9f0441151fd8d38b77597b71e5742419374b0` | `09c4ccbe6521f863cd1deb2e55de2ad6a7c570763510afe42b368fe52cd85d81` | `91042b29a17e22d1c92e41f4f9e18a6d96651b88a00bd29c48b0bf23ebe10aea` |

The same proposed Bernoulli field is shared across all 26 policy arms within
each of 72 model-level fields (24 per BER). Methods are random, MSB-first,
numerical bit jump, global decoded-code surrogate gradient, per-example absolute
surrogate gradient, and per-example positive surrogate gradient. Each method has
6.25%, 12.5%, 18.75% and 25% exact-bit interior budgets. Zero and full protection
are shared controls, not additional independent policy replicates. The three
expected proposed counts are 4, 32, 256. At each count, expected applied counts
for 0%, 6.25%, 12.5%, 18.75%, 25%, 100% coverage are respectively
`λ × {1, .9375, .875, .8125, .75, 0}`. Each field scores 4,096 WikiText-2 test
targets; eight disjoint validation windows construct the gradient rankings.
Primary loss is teacher-forced target-token NLL, capped at the frozen 30-nat
numerical-failure penalty. Failure fraction and perplexity are separate outcomes.

Pythia has 988,545,024 eligible INT8 code bits in 49 tensors. Its BERs are
`4.046350851896049e-9`, `3.237080681516839e-8`, and `2.5896645452134713e-7`.
Qwen has 4,767,875,072 eligible bits in 197 tensors including its separately
stored output-head copy. Its BERs are `8.389481560644381e-10`,
`6.7115852485155046e-9`, and `5.3692681988124037e-8`. Scales, embeddings,
normalization, biases and activation codes remain outside the faultable universe.
Neither study is whole-model storage robustness. FKeras and PrioriFI remain
unmeasured, without substitute rankings.

## Resource admission and current execution

The frozen v1 gate admitted only one model at a time. On 2026-11-04, M11 was
absent but two M12 matched reliability jobs were active at about one CPU core
each. Versioned v3 permits co-execution only when measured major-model CPU use
is at most four cores total, each major model RSS is at most 16 GiB, host busy
fraction is at most 0.5, runnable/load limits pass, and the new campaign can
claim four distinct physical cores. The new process is pinned to those cores and
monitored against the original per-model RAM caps. The gate preserves one W8A8
LM campaign at a time and does not modify existing jobs.

Pythia admission measured 7.28% host CPU busy, one-minute load 4.47,
423.02 GiB available RAM and 17,100.28 GiB free disk. Its new affinity is
CPUs `2,4,6,8`; the write-once admission receipt SHA-256 is
`6b7ec369fe20dd675d705df8f949cdcccc137dfef2b94f987f0d324e77d7889e`.
It launched in tmux session `netsurf-pythia-w8a8-v3`. The budget is four CPU
threads, 12 GiB RSS and 10 GiB free output disk. The original 8-hour estimate
is provisional. The streamed ranking requires 16.805 billion score-key visits,
0.989 GB of two gradient-moment streams, and 2.471 GB of packed interior masks.
It is an actual full eligible weight campaign in progress, not finalized evidence.

Qwen's frozen scientific v3 plan passes validation, but Qwen has **not** been
launched. The sequential rule blocks it while Pythia is active. Its pre-run
allowances are four CPU threads, 64 GiB RSS, 40 GiB free disk and a provisional
24 hours. Its ranking requires 81.054 billion score-key visits, 4.768 GB of
gradient moments and 11.920 GB of packed masks. The full selection and final
cost may exceed that estimate; do not change its scientific grid after seeing
Pythia or Qwen effects.

## Foreground commands and exports

The active Pythia pane and log can be inspected with:

```sh
tmux attach -t netsurf-pythia-w8a8-v3
tail -f /projects/netsurf/manuelbv/workspace/implementation-worktrees/transformer-integration/workspace/logs/pythia160m_w8a8_reliability_v3_run1.log
```

The Pythia wrapper automatically validates, selects, seals, evaluates, analyzes,
finalizes a complete scientific ZIP, and writes a runtime receipt. Once that
foreground process exits successfully, produce the additional condition CSV,
field CSV, loss-versus-BER plot and compact export with this read-only offline
command (it verifies the completed core before writing a separate report):

```sh
cd /projects/netsurf/manuelbv/workspace/implementation-worktrees/transformer-integration
/projects/kona-assets/manuelbv/envs/torch-2.9.1-cpu-py311/bin/python scripts/report_lm_w8a8_v3.py \
  /projects/netsurf/manuelbv/workspace/implementation-worktrees/transformer-integration/workspace/plans/pythia160m_w8a8_reliability_v3 \
  /projects/netsurf/manuelbv/workspace/implementation-worktrees/transformer-integration/workspace/runs/pythia160m_w8a8_reliability_v3_run1
```

After Pythia's core ZIP is finalized and verified, the researcher can launch
the complete Qwen foreground continuation in its own persistent tmux session.
It verifies Pythia finalization, rechecks live Qwen resources, runs the frozen
Qwen phases, and then creates Qwen's offline CSV/plot/compact export:

```sh
tmux new-session -s netsurf-qwen-w8a8-v3 'bash -o pipefail -c "/projects/netsurf/manuelbv/workspace/implementation-worktrees/transformer-integration/scripts/run_qwen_w8a8_v3_after_pythia.sh 2>&1 | tee /projects/netsurf/manuelbv/workspace/implementation-worktrees/transformer-integration/workspace/logs/qwen3_0_6b_w8a8_reliability_v3_run1.log"'
```

Each finalized LM report writes `mac_scp_commands.txt` with exact Mac commands.
The Pythia destinations, after completion, are:

```sh
scp kona:/projects/netsurf/manuelbv/workspace/implementation-worktrees/transformer-integration/workspace/runs/pythia160m_w8a8_reliability_v3_run1_export/complete.zip ~/Downloads/pythia160m_int8_w8a8_complete.zip
scp kona:/projects/netsurf/manuelbv/workspace/implementation-worktrees/transformer-integration/workspace/runs/pythia160m_w8a8_reliability_v3_run1_reporting_v3/compact.zip ~/Downloads/pythia160m_int8_w8a8_compact_v3.zip
scp kona:/projects/netsurf/manuelbv/workspace/implementation-worktrees/transformer-integration/workspace/runs/pythia160m_w8a8_reliability_v3_run1_reporting_v3/receipt.json ~/Downloads/pythia160m_int8_w8a8_reporting_receipt_v3.json
```

The Qwen destinations are:

```sh
scp kona:/projects/netsurf/manuelbv/workspace/implementation-worktrees/transformer-integration/workspace/runs/qwen3_0_6b_w8a8_reliability_v3_run1_export/complete.zip ~/Downloads/qwen3_0_6b_int8_w8a8_complete.zip
scp kona:/projects/netsurf/manuelbv/workspace/implementation-worktrees/transformer-integration/workspace/runs/qwen3_0_6b_w8a8_reliability_v3_run1_reporting_v3/compact.zip ~/Downloads/qwen3_0_6b_int8_w8a8_compact_v3.zip
scp kona:/projects/netsurf/manuelbv/workspace/implementation-worktrees/transformer-integration/workspace/runs/qwen3_0_6b_w8a8_reliability_v3_run1_reporting_v3/receipt.json ~/Downloads/qwen3_0_6b_int8_w8a8_reporting_receipt_v3.json
```

These LM export paths are prospective until their `FINALIZED` files exist.

DistilBERT v2 is already finalized and independently reverified. Its separate
M12 handoff is `config/w8a8_m12_handoff_distilbert_v2.json`. M12 owns database
ingestion and must verify the full ZIP, compact ZIP, hashes and 78 condition rows
before adding it as a distinct W8A8 study. No M12 database file was changed.
