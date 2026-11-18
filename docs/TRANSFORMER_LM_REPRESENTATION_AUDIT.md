# Targeted LM representation and numerical-failure audit

## Evidence and interpretation

This read-only audit reconstructs the three completed 96-bit studies from their immutable complete ZIPs. Its auditor checks archive CRC and checksums, frozen plan/source/asset hashes, selection seals, all 96 exact-isolated response-file hashes per model, final intents and schedules, policy event accounting, and every saved token's logit statistics against the saved mean NLL. The new derived bundle is `workspace/runs/lm_representation_offline_audit_v1`, payload SHA-256 `5c430f63fc31bbc4297116cceb4f58afc2d1cf12f49371afad60cd3e15e75bc3`, complete ZIP SHA-256 `973f88ea783fc82703ef9bb10dfdb02aa7d68b19d62c8bf6dc59cb5cae3e5dc7`, `FINALIZED` digest `7379ee0f9322564c8000488b16bdad8b01df5609e2919eedaef0ef920f535052`. The original complete ZIP SHA-256 values remain Pythia FP32/FP32 `a1bde2b1a68df8c15607eb04a9c05b715b1009ee6d399301031cb5211c829b91`, Pythia BF16/FP32 `a51fe02c0ea903ff44b97ef846ab246e9ea20cd8867c1662fd9c4ddcad903345`, and Qwen BF16/BF16 `5fa889ae32e7fc20132ea09f90476b3f672ca4006406d0e09dcef5c43c2abb62`. No model inference was repeated in this audit.

Every panel contains exactly 32 sign, 32 highest-exponent and 32 high-significand coordinates, one of each from 32 stored words. At budgets 12 and 24, all significance-selected bits are sign bits and all exact-isolated-selected bits are highest-exponent bits; their intersections are zero. The original Pythia FP32 and BF16 studies have identical selected word ordinals, normalized sign/exponent/significand masks and normalized per-field fault coordinates and seeds. The encoded bit indices, source bytes and execution representations differ. They are a paired precision diagnostic, not independent model replications.

The new **exponent-first structural baseline** ranks exponent class above sign above significand and breaks every within-class tie by parsed `(tensor name, scalar-word ordinal, bit index)`. It uses no labels, gradient or observed fault outcome. It was fixed after the original final results, so its follow-up is explicitly prompted by those results. The complete eight-mask overlap matrices are in `audit.json`; key intersections are:

| Target | Budget | Significance ∩ isolated | Exponent-first ∩ isolated | Exponent-first ∩ significance |
|---|---:|---:|---:|---:|
| Pythia FP32/FP32 | 12 | 0 | 11 | 0 |
| Pythia FP32/FP32 | 24 | 0 | 23 | 0 |
| Pythia BF16/FP32 | 12 | 0 | 11 | 0 |
| Pythia BF16/FP32 | 24 | 0 | 23 | 0 |
| Qwen BF16/BF16 | 12 | 0 | 7 | 0 |
| Qwen BF16/BF16 | 24 | 0 | 17 | 0 |

Because no exponent-first mask equals the corresponding exact-isolated mask, the old final records cannot be relabeled as new exponent-first outcomes. A fresh paired study is frozen below. In Pythia, 31 exponent coordinates have an identical exact-isolated score: FP32 `26.4089235321` and BF16/FP32 `26.4093426081`. Each equals **30 minus the corresponding clean calibration NLL**, establishing score saturation at the prespecified 30-nat loss cap. Both budget boundaries cut through this 31-way tie; canonical tie order accounts for the one-bit exponent-first/isolated difference at each budget. Qwen's largest score is `9.4136381825`, below its possible cap difference `27.4145328792`; 13 scores tie at `9.3457472118`, crossing the 12-bit boundary, while its 24-bit boundary is not tied. The Qwen tie is observed, but is not explained by cap saturation.

## Numerical failure reconstruction

The table concerns **BER 4/96**, 32 paired master fields, 256 scored tokens per field and the two primary budgets. `Full` means all 256 tokens have nonfinite logits or inference failure; `partial` means some failed tokens; `finite cap` means at least one finite token loss reaches 30 nats without a failed token. None of the rows below had partial fields or finite-capped tokens. Failure-penalty and ordinary-finite columns sum to saved mean NLL, up to floating-point reduction tolerance.

| Model, budget | Policy | Full / partial / finite-cap / ordinary fields | Failed tokens / 8,192 | Fields with applied highest exponent | Failure-penalty NLL | Ordinary-finite NLL | Total NLL |
|---|---|---|---:|---:|---:|---:|---:|
| Pythia FP32, 12 | Significance | 24 / 0 / 0 / 8 | 6,144 | 24 | 22.5000 | 0.9198 | 23.4198 |
| Pythia FP32, 12 | Isolated | 23 / 0 / 0 / 9 | 5,888 | 23 | 21.5625 | 1.0348 | 22.5973 |
| Pythia FP32, 24 | Significance | 24 / 0 / 0 / 8 | 6,144 | 24 | 22.5000 | 0.9199 | 23.4199 |
| Pythia FP32, 24 | Isolated | 16 / 0 / 0 / 16 | 4,096 | 18 | 15.0000 | 1.8445 | 16.8445 |
| Pythia BF16/FP32, 12 | Significance | 24 / 0 / 0 / 8 | 6,144 | 24 | 22.5000 | 0.9206 | 23.4206 |
| Pythia BF16/FP32, 12 | Isolated | 23 / 0 / 0 / 9 | 5,888 | 23 | 21.5625 | 1.0357 | 22.5982 |
| Pythia BF16/FP32, 24 | Significance | 24 / 0 / 0 / 8 | 6,144 | 24 | 22.5000 | 0.9206 | 23.4206 |
| Pythia BF16/FP32, 24 | Isolated | 16 / 0 / 0 / 16 | 4,096 | 18 | 15.0000 | 1.8461 | 16.8461 |
| Qwen BF16, 12 | Significance | 2 / 0 / 0 / 30 | 512 | 24 | 1.8750 | 7.3752 | 9.2502 |
| Qwen BF16, 12 | Isolated | 0 / 0 / 0 / 32 | 0 | 22 | 0 | 5.7878 | 5.7878 |
| Qwen BF16, 24 | Significance | 2 / 0 / 0 / 30 | 512 | 24 | 1.8750 | 7.3745 | 9.2495 |
| Qwen BF16, 24 | Isolated | 0 / 0 / 0 / 32 | 0 | 9 | 0 | 2.8610 | 2.8610 |

For Pythia at budget 24, the **7.5-nat** difference in failure-penalty contribution exceeds the observed roughly **6.575-nat** total contrast, offset by about −0.925 nats of ordinary finite loss. Thus the failure policy explains the mean contrast's direction and more than its magnitude; the finite component partly offsets it. Applied highest-exponent faults align perfectly with complete nonfinite fields for Pythia significance, but not for Pythia isolated at budget 24: 18 fields have such a fault and only 16 fail. For Qwen significance, 24 fields have an applied highest-exponent fault and only two fail; most of its total NLL contrast is finite loss. These are **associations of multi-fault schedules**, not causal attribution to an individual bit. The saved exact one-bit calibration responses establish single-bit harm on that calibration cohort, but multi-fault causality needs fresh counterfactual controls.

The original paired field confidence intervals, Holm families, favorable/adverse/tied counts and heavy-tail diagnostics remain in the original immutable analyses and [manuscript evidence table](TRANSFORMER_MANUSCRIPT_INTEGRATION.md). The derived audit adds 512 field-policy rows per model across both nonzero BER arms, with proposed/applied event counts, applied highest-exponent counts and per-token failure decomposition. No new p-value is presented as confirmatory.

## Frozen fresh-window structural controls

Three hash-bound, model-specific control plans now bind the old panel and selection masks, the data-free exponent-first masks, verified READY assets, source, independent new fault seeds and **fresh test token IDs 257–513 as input, scoring target IDs 258–513**. The original pilot scored test targets 1–256. Each control plan evaluates random, sign-first, exponent-first and exact isolated at budgets 12 and 24, plus clean/unprotected/full-panel controls, under 32 paired fields at each BER 1/96 and 4/96. The primary comparison is exponent-first minus isolated at BER 4/96 for both budgets, two-sided Holm across two; a distinct twelve-comparison secondary Holm family is frozen. The old selection masks are reused without re-selection. These are prospective follow-ups motivated by known pilot outcomes, not untouched discovery hypotheses. The same Pythia scalar ordinals and field seeds intentionally preserve their paired precision diagnostic. Each plan has **641 final model requests and 164,096 scored-token requests**. The scripts stop while the two M11 CPU workers are active; no control final inference has run.

The original 64-field final phases spanned about 37 minutes for Pythia FP32, 49 minutes for Pythia BF16/FP32 and 28 minutes for Qwen on the shared host, measured from first to last saved field. The controls have the same field count but ten policy calls per field versus eight previously. Provision **two foreground hours and 4 GiB RAM per Pythia control**, and **two hours and 6 GiB RAM for Qwen**, one target at a time. These are operational allowances, not a throughput guarantee; faulted numerical behavior and concurrent load can change them.

| Target | Frozen plan object SHA-256 | Frozen plan file SHA-256 |
|---|---|---|
| Pythia FP32/FP32 | `4e001afe269ea058a8ae9aea9332c3056cd5bfbc9f3af5e07850200174070d6f` | `39617c4c0c2afa90cac0abe8c5ecccf79e2403732c31be3bcdb17cf26fee464d` |
| Pythia BF16/FP32 | `9bb5b9b201f9bca1469c1c8d98ffbb60d9070e65f54adfe52a1860a4151b5f8d` | `b973516fc40dadcb3f0888545aade0238ceb60c1ed1d6b5a0e426fed3407e52f` |
| Qwen BF16/BF16 | `6a9d149634e4544fc512dee28f65fcfe7a551eda3c9e7326ec371eb672bc9771` | `fde6a00f7169e05b28bff9d439b5c7d8b5f78fec6bebfa788b4c70db38174849` |

The frozen plans live under `workspace/plans/{pythia160m_fp32,pythia160m_bf16fp32,qwen3_0_6b_bf16}_exponent_control_v1/plan.json`. The one-shot foreground runner is `scripts/run_lm_exponent_control_once.sh`; it validates assets, runs all paired fields, aggregates offline, creates a complete ZIP and prints the Mac `scp` command. Run roots are intentionally absent. Historical 96-bit ZIPs and masks are unchanged. A Qwen full-text tokenization warning is expected: actual inference uses only the 257-token frozen slice, within the validated context.

After the M11 workers finish, run **one target at a time** in the Kona foreground. Each line below is one complete command; its asset snapshot is the frozen READY copy from that target's original pilot:

```sh
<repo>/workspace/implementation-worktrees/transformer-integration/scripts/run_lm_exponent_control_once.sh pythia160m_fp32stored_fp32compute <repo>/workspace/implementation-worktrees/transformer-integration/workspace/runs/pythia160m_fp32_targeted96_v1_run1_export/complete.zip <repo>/workspace/implementation-worktrees/transformer-integration/workspace/plans/pythia160m_fp32_targeted96_v1/NETSURF_ASSET_READINESS.json <repo>/workspace/implementation-worktrees/transformer-integration/config/transformer_runtime_registry.json <repo>/workspace/implementation-worktrees/transformer-integration/workspace/plans/pythia160m_fp32_exponent_control_v1 <repo>/workspace/implementation-worktrees/transformer-integration/workspace/runs/pythia160m_fp32_exponent_control_v1_run1
<repo>/workspace/implementation-worktrees/transformer-integration/scripts/run_lm_exponent_control_once.sh pythia160m_bf16stored_fp32compute <repo>/workspace/implementation-worktrees/transformer-integration/workspace/runs/pythia160m_bf16fp32_targeted96_v1_run1_export/complete.zip <repo>/workspace/implementation-worktrees/transformer-integration/workspace/plans/pythia160m_bf16fp32_targeted96_v1/NETSURF_ASSET_READINESS.json <repo>/workspace/implementation-worktrees/transformer-integration/config/transformer_runtime_registry.json <repo>/workspace/implementation-worktrees/transformer-integration/workspace/plans/pythia160m_bf16fp32_exponent_control_v1 <repo>/workspace/implementation-worktrees/transformer-integration/workspace/runs/pythia160m_bf16fp32_exponent_control_v1_run1
<repo>/workspace/implementation-worktrees/transformer-integration/scripts/run_lm_exponent_control_once.sh qwen3_0_6b_bf16 <repo>/workspace/implementation-worktrees/transformer-integration/workspace/runs/qwen3_0_6b_bf16_targeted96_v1_run1_export/complete.zip <repo>/workspace/implementation-worktrees/transformer-integration/workspace/plans/qwen3_0_6b_bf16_targeted96_v1/NETSURF_ASSET_READINESS.json <repo>/workspace/implementation-worktrees/transformer-integration/config/transformer_runtime_registry.json <repo>/workspace/implementation-worktrees/transformer-integration/workspace/plans/qwen3_0_6b_bf16_exponent_control_v1 <repo>/workspace/implementation-worktrees/transformer-integration/workspace/runs/qwen3_0_6b_bf16_exponent_control_v1_run1
```

Once a given command has finalized its bundle, copy its complete ZIP to the Mac with the corresponding command printed and saved by the runner. The paths are:

```sh
scp kona:<repo>/workspace/implementation-worktrees/transformer-integration/workspace/runs/pythia160m_fp32_exponent_control_v1_run1_export/complete.zip ~/Downloads/pythia160m_fp32stored_fp32compute_exponent_control_complete.zip
scp kona:<repo>/workspace/implementation-worktrees/transformer-integration/workspace/runs/pythia160m_bf16fp32_exponent_control_v1_run1_export/complete.zip ~/Downloads/pythia160m_bf16stored_fp32compute_exponent_control_complete.zip
scp kona:<repo>/workspace/implementation-worktrees/transformer-integration/workspace/runs/qwen3_0_6b_bf16_exponent_control_v1_run1_export/complete.zip ~/Downloads/qwen3_0_6b_bf16_exponent_control_complete.zip
```

The completed **offline audit** bundle can be copied now:

```sh
scp kona:<repo>/workspace/implementation-worktrees/transformer-integration/workspace/runs/lm_representation_offline_audit_v1/complete.zip ~/Downloads/lm_representation_offline_audit_v1_complete.zip
```
