# Transformer storage and benchmark protocol

**Subsequent status, 2 November:** the researcher has launched the frozen
DistilBERT study in its original `transformer-integration` worktree. The
historical "not running" statements below describe its prelaunch handoff, not
current process status. Do not edit or restart that study. Its engineering
receipt was reverified with payload
`62a379160e207fb9041451e28b9e6d76ae44646ae4291cfa4e8e95c2afc877cf`;
no unfinalized campaign outcomes are accepted as manuscript results.

Pythia now has separate FP32 and BF16-storage/FP32-computation clean reports;
the latter is awaiting a distinct finalized READY record. Native BF16 remains
VALIDATION_FAILED. This branch prepares and synthetically tests the mixed
storage/computation path only; see `PYTHIA_BF16_STORAGE.md`. Qwen Base remains
under validation and must not be treated as the older post-trained portfolio
checkpoint. No shared assets or existing source pins were changed.

**2 November 2026.** DistilBERT/SST-2 is `READY` in Kona's verified asset manifest and has passed a 32-example real-model storage-fault conformance trial. Its fault/protection results are **engineering only**. No transformer BER or protection-budget effectiveness campaign has run. Pythia and Qwen BF16 are still `PREPARED_AWAITING_APPROVAL`; their model assets are not executable NetSuRF evidence. The three validated TinyML benchmarks remain the ACM TECS manuscript evidence backbone.

## DistilBERT source and executable boundary

The Kona `READY` record pins `distilbert/distilbert-base-uncased-finetuned-sst-2-english` at commit `714eb0fa89d2f80546fda750413ed43d93601a13`, FP32 safetensors checkpoint SHA-256 `7c3919835e442510166d267fe7cbe847e0c51cd26d9ba07b89a57b952b49b8aa`, tokenizer hashes, all three SST-2 Parquet hashes, an environment lock hash, and the clean-reference report hash. The consumer verifies every byte before model loading, including Hugging Face snapshot symlinks into that repository's `blobs/` tree. It executes the manifest's absolute environment entrypoint, Python 3.11.18, PyTorch 2.9.1+cpu, Transformers 4.57.1, eight CPU threads, deterministic algorithms and eager attention. The loader uses local files only, eval mode, no autocast, no GPU and no stochastic generation. Every inspected eligible runtime tensor's dtype, shape and bytes must match its safetensors source.

The full **inspectable** fault universe is contiguous, non-aliased `torch.nn.Linear.weight` FP32 parameters: **1,377,878,016 encoded bits** in this DistilBERT executable. Embeddings, layer normalization, biases, activations and caches are excluded. Tied/aliased linear weights are explicitly listed and excluded rather than counted twice. `stored-xor/v1` performs exact little-endian XOR of sign, exponent or significand bits in the live parameter's stored bytes; it does not approximately modify a float. A trial preserves those mutated bytes across the complete input cohort. The backend checks the exact runtime word before inference, checks pointers, restores every original byte after the trial and verifies storage hashes. Protected coordinates suppress events logically; this is neither physical ECC nor a physical SEU model. The original INT8 TFLite path is unchanged.

## Real-model engineering evidence

The write-once receipt is [DistilBERT conformance v3](../workspace/runs/distilbert_sst2_fp32_conformance_v3/manifest.json). It binds committed source `ca2bd69e6d57e99e7ab80feaa4384533e5aad66e`, the READY manifest and target record, 32 public-labeled SST-2 validation rows, all cohort identities and digests, exact forced event lists, schedule and mask hashes, clean/faulted logits, losses, accuracy, failed examples, byte-exact restoration, elapsed time and checksums. The source was clean at execution.

On those 32 engineering rows, clean mean cross-entropy was **0.15122652 nats** and accuracy **31/32**. A classifier sign-bit fault changed mean loss to **0.14985874**; an exponent bit to **0.15088426**; and a high significand bit to **0.15151826**. These signs are retained: a fault may help or harm this particular cohort. Exactly encoded infinity and quiet NaN each caused 32/32 nonfinite failures and the prespecified **30-nat** penalty per example. The same two-bit field yielded mean loss **0.15007465** unprotected and **0.15144299** when one coordinate was protected. Full suppression reproduced clean logits and loss exactly. All trials restored original stored bytes. Total trial wall time was **32.35 s** and peak RSS **823.3 MB**. These forced tests establish actual bit consumption and task-level protection execution; they are not random-BER effectiveness evidence. The earlier v2 receipt remains immutable and is superseded by the committed-source v3 receipt.

Kona's clean reference evaluated all 872 labeled validation examples before this study. That was **clean inference exposure**; it is documented, not erased. NetSuRF fault outcomes have been observed only for validation rows 0–31. The held-out fault-outcome set begins at validation row 32. The GLUE public test split has withheld labels and is never treated as labeled evaluation data.

## Frozen targeted study

The first production candidate uses **only the first 32 scalar words of `classifier.weight`, all 32 FP32 bits per word**: 1,024 prespecified bit coordinates. This fixed tensor-order rule does not use engineering fault outcomes or select favorable effects. It is a targeted classifier-bit fault model, not an estimate of whole-transformer reliability. The other 1,377,876,992 inspectable linear-weight bits are outside this study's fault-law support. Its candidate list and hashes are in the frozen JSON plan under `config/experiments/`.

The plan reserves Kona-hash-ranked SST-2 train rows as 512 calibration and 128 distinct model-selection examples. It uses the first 32 selection examples for exact isolated-bit ranking, then holds the remaining 96 selection rows unused. Engineering uses validation rows 0–31; final uses validation rows 32–287 (256 labeled clips); rows 288–871 stay unused for future fault-outcome work. The model was fine-tuned on train, so selection data are not independent of model training; final validation fault outcomes remain disjoint. Complete identities and record hashes are frozen. All three BER arms share the same final clips and their dependence is recorded.

The BER grid is `p = 0, 1/1024, 4/1024, 16/1024`, chosen as **0, 1, 4 and 16 expected proposed faults** over the declared 1,024-bit support. For each nonzero BER there are 32 independent persistent Bernoulli fields with fresh fixed seeds. Each field is reused across all protection policies. Budgets are `B = 0, 128, 256, 1024`; expected applied faults are `(1024-B)p`. The methods are seeded random, encoded-bit significance ordering and exact empirical isolated-bit selection on the 32 selection clips. Exact isolated enumeration runs all 1,024 single-bit states and retains signed score values. There is no guided optimizer. The full-budget policy is a clean restoration control. The 128- and 256-bit masks are selected and sealed before any final field runs.

The primary field-level contrast is exact isolated versus random at the two intermediate budgets and three nonzero BERs. Exact isolated versus significance is secondary. The prespecified **12-comparison family** uses paired field-level sign-flip tests with Holm adjustment; 10,000 field bootstrap replicates give percentile intervals. The sign-flip test assumes a symmetric null distribution. Report mean cross-entropy, accuracy, failed clip fraction, expected applied faults, favorable/adverse field counts, median and maximum absolute field differences. Fields are independent stochastic units; 256 clips within a field are paired repeated measurements, not 256 independent fault replicates. Reusing clips across BERs induces cross-arm dependence. Fixed cohort analysis does not estimate uncertainty from sampling a new SST-2 input population. Heavy tails and nonfinite outcomes remain visible under the 30-nat capped-risk rule.

The frozen cost is **1,025 selection model requests / 32,800 clip inferences** and **769 final model requests / 196,864 clip inferences**: **1,794 requests / 229,664 clip inferences** total. With batches of eight, this is 28,708 PyTorch forwards. The engineering throughput implies about **5.86 foreground hours**; this is a planning estimate, not a power calculation. Measured engineering peak RSS was 823.3 MB. The complete study has **96 independently seeded final fields**, eight unique protection policies per field, and a 128-event fail-closed per-field cap that never truncates a draw. Low-BER contrasts may be unresolved; the field count will not be increased after seeing favorable or adverse final effects.

Earlier frozen plan versions remain preserved. Versions 1–3 were invalidated before selection by the environment-entrypoint fix, language-model adapter completion, and a stronger offline schedule-replay gate. None has selection or final exposure. Version 4 is authoritative and source-hash bound; older versions fail closed. No selection or final stage has been launched automatically.

## Foreground execution and evidence return

The committed-source engineering receipt can be verified and exported with one foreground command:

```bash
<repo>/workspace/implementation-worktrees/transformer-integration/scripts/run_distilbert_engineering_once.sh \
  <repo>/workspace/implementation-worktrees/transformer-integration/workspace/plans/distilbert_sst2_classifier1024_v4/NETSURF_ASSET_READINESS.json \
  <repo>/workspace/implementation-worktrees/transformer-integration/workspace/runs/distilbert_sst2_fp32_conformance_v3
```

The paired targeted study is deliberately **not running**. Its single foreground Kona command validates the frozen plan, seals every selection member, runs every final member with immutable same-seed resume, aggregates offline, creates CSV/SVG analysis, makes compact and complete ZIPs, and saves/prints exact Mac `scp` commands:

```bash
<repo>/workspace/implementation-worktrees/transformer-integration/scripts/run_distilbert_targeted_campaign_once.sh \
  <repo>/workspace/implementation-worktrees/transformer-integration/config/experiments/distilbert_sst2_classifier1024_v4.json \
  <repo>/workspace/implementation-worktrees/transformer-integration/workspace/plans/distilbert_sst2_classifier1024_v4/NETSURF_ASSET_READINESS.json \
  <repo>/workspace/implementation-worktrees/transformer-integration/workspace/runs/distilbert_sst2_classifier1024_v4
```

Selection-only is the same interpreter running `python -m netsurf.transformer_campaign_cli selection PLAN READINESS RUN_ROOT`. The `final` phase refuses to start without a plan-bound seal for all three methods. `aggregate` and `export` work offline on finalized members. A partial member is retained and rejected; no replacement seed or sample identity is silently generated. ZIPs and `mac_scp_commands.txt` appear next to the completed run. The researcher can inspect the plan and cost before starting the long foreground command.

## Pythia and Qwen feasibility

The teacher-forced adapter supports exact single-pass tokenization, 1,024-token contexts, stride 512, no BOS/EOS, joined WikiText-2 raw rows, and one score per token after position zero. It records token identity, target logit and log-sum-exp, token NLL, perplexity and failure fraction. A deterministic synthetic PyTorch fixture has passed; this is **not** Pythia or Qwen reliability evidence. The actual BF16 targets remain unavailable for NetSuRF execution until Kona marks each record `READY`, verified source revisions match the NetSuRF portfolio, and the exact runtime dtype and stored-byte restoration gate pass. Pythia's proposed BF16 artifact is derived from FP32 and requires its own approved hash and clean reference. Qwen's tied output weight is excluded explicitly by the alias-safe linear-weight population. On this CPU BF16 is not natively accelerated, so measure real-window throughput and peak RSS before freezing a large campaign. A distinct FP32 computation representation, if needed, requires a separate storage-to-computation fault contract and conformance; it cannot be relabeled BF16 execution.

For any new model, calculate `lambda=(N-B)p` using its verified eligible bit count before proposing BERs. Equal BER or equal lambda across precisions has no physical reliability equivalence. INT8/INT4 transformer targets remain blocked until scale, zero point, grouping, packed-code ordering and actual inference consumption are verified. Gradient ranking primitives exist but no transformer autograd extraction or numerical Hessian-vector-product baseline is validated. FKeras/Hessian, qPolarGrad reinterpretation and a new optimizer are absent.

## Manuscript boundary

The status ledger `config/transformer_target_status.json` holds all 18 targets with evidence paths and hashes. DistilBERT may be described as a passed engineering backend and a frozen prospective study, with **no effectiveness point or curve yet**. Add a transformer benchmark row and BER/budget figures to the November 9 ACM TECS draft only after the paired result bundle is finalized and checksum-verified. Keep accuracy, ROC-AUC and perplexity on separate numerical axes. Do not modify the historical TinyML, M11 or asset evidence.
