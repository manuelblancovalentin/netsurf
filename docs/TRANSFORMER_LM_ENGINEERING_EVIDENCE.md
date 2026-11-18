# Pythia and Qwen: pinned real-fault engineering evidence

All three records below are **finalized engineering smokes**, not BER-versus-reliability or publication-scale protection results. The source was commit `33518368311d4a6f3ce7542667993e60fa895893`; the Kona READY manifest snapshot was SHA-256 `3192d3ce57b7a11089af4c1e0f802e851ebc8b179fd823a793fecf333313db35`. Source, asset and schedule hashes are stored in each complete bundle. The original `NETSURF_ASSET_READINESS.json` and its approved assets were not edited by NetSuRF. Native-BF16-computation Pythia remains `VALIDATION_FAILED` and was never substituted for either READY runtime.

## Executed bounded trials

Each target used the first 33 token IDs of the pinned WikiText-2 validation sequence, scoring 32 targets once by deterministic teacher forcing. Every trial used the same pinned tokenizer, 1,024-token maximum context, 512-token stride, CPU eager attention and evaluation mode. Each smoke made ten model requests: clean twice, sign, exponent, significand, two-bit unprotected, the same two-bit field with exponent protection, full protection, repeated unprotected field, and clean after restoration. The deliberately forced three-bit support is a diagnostic subset of the verified `nn.Linear.weight` universe. It is **not** a full-model Bernoulli BER test. Failure/penalty semantics are 30 nats per failed target token, with the failed count reported separately. The same schedule and unprotected risk reproduced exactly; full suppression reproduced clean risk. Every stored parameter was restored byte-exactly. For Pythia BF16 storage with FP32 computation, the BF16 source word changed by encoded XOR before an independently allocated FP32 copy consumed the widened value; source and execution copies were both restored and verified.

| Runtime configuration | Full eligible linear-weight bits | Clean mean token NLL | Sign / exponent / significand NLL | Paired two-bit unprotected → exponent-protected NLL | Failed targets under exponent | Runtime / peak RSS |
|---|---:|---:|---|---|---:|---|
| Pythia FP32 stored, FP32 compute | 3,954,180,096 | 4.13310 | 4.12395 / 29.33874 / 4.13917 | 29.33874 → 4.12395 | 5/32 | 31.08 s / 1,362 MiB |
| Pythia BF16 stored, FP32 compute | 1,977,090,048 | 4.14213 | 4.13303 / 29.33951 / 4.14815 | 29.33951 → 4.13303 | 5/32 | 37.88 s / 1,984 MiB |
| Qwen BF16 stored, BF16 compute | 7,046,430,720 | 3.41477 | 3.42542 / 11.93121 / 3.43158 | 11.93121 → 3.42542 | 0/32 | 15.00 s / 2,044 MiB |

The paired protected/unprotected outcomes demonstrate that a real stored-bit fault reached task inference and that logical suppression worked on this small, selected field. They do not estimate average protection effectiveness. The clean 32-token NLLs are local controls, not the asset agent's full-test reference perplexities.

## Scientific identity and archive receipts

| Target ID | READY record SHA-256 | Payload SHA-256 | Complete ZIP SHA-256 | Compact ZIP SHA-256 |
|---|---|---|---|---|
| `pythia160m_fp32stored_fp32compute` | `246b5079221b6e5e3267d97a4514c24de3c5fb945620ac3d160e2bacd238863b` | `1556692a4eecdeb4e0f6899421709ce5eb7672f710df0354c93f2e3a385fc7fa` | `0c66c6aba43e05566b98bc5f5276f195c2688b11c30147fdaa2b321485a1561f` | `7ec81c368b8bc8f736925e050f8aa645e92a2cb96b538e62936dd8f1d41112f4` |
| `pythia160m_bf16stored_fp32compute` | `4e2207e2a8dd91619635e2198a0af514b35e08497392f7df0324098b71cd0ec5` | `13fc5782437fbcdcda19bb9c3c4a9f2d4ba2bf5c523a779c772f1610cfe8f07c` | `fbac5823eedb5abcf3585ff08479b2b251db9da560e5f79d920dfcf87b80b393` | `f24d45af83a8d834477a7a50f356245f404fe343e6e05199fbc813229225bb6d` |
| `qwen3_0_6b_bf16` | `1e8448bb03baf060f1a8c403584d3b5edd91dad1d5c449a3a4ea68e627381106` | `d29bbea1dc2b99c35cca9429981219ea4244b223f385bfc5901b1e2d9e21d446` | `ad8bfdce29349cca37af304b306d6ed85fbb1fc2dc9acf275bd3058c4347e813` | `7a68b8a3398c41bc7829ec4d380c5383cdd1f43a0c1d6cbc3d96dd9489d9ba99` |

All eight listed artifact checksums per run matched. All three complete ZIPs passed CRC checks; each payload digest matched its `FINALIZED` marker. Each bundle retains its exact Mac `scp` commands in `mac_scp_commands.txt`.

## Full-universe Bernoulli feasibility before a production plan

The exact initial fault law for a future full-linear-weight study would draw independent Bernoulli events across the listed eligible bit universe, with `lambda = (N-B)p` after a protection budget `B`. Equal BER has dramatically different absolute exposure across these targets. For the **unprotected** universe, the following BERs would yield only the indicated expected fault counts per persistent trial; these are arithmetic feasibility proposals, not frozen conditions or results.

| Target | `p` for λ=0.25 | `p` for λ=1 | `p` for λ=4 |
|---|---:|---:|---:|
| Pythia FP32/FP32 | `6.32242320609e-11` | `2.52896928244e-10` | `1.01158771297e-9` |
| Pythia BF16/FP32 | `1.26448464122e-10` | `5.05793856487e-10` | `2.02317542595e-9` |
| Qwen BF16/BF16 | `3.54789552234e-11` | `1.41915820894e-10` | `5.67663283575e-10` |

A protected count `B` changes the expected applied count to `(N-B)p`; every future table must report both BER and absolute proposed/applied exposure. These levels must not be selectively pruned after looking at outcomes, and equal λ does not imply equivalent physical reliability or equivalent damage. The current sampler can generate exact Bernoulli fields over the full eligible population without a billions-entry bit vector, with a fail-closed event cap. It has not yet been calibrated for an affordable, informative **full-model** protection allocation. A 24-bit or 128-bit mask in a multi-billion-bit universe would suppress almost no randomly sampled events at these λ values; a large mask requires a validated scalable selector and memory/cost accounting. A publication-ready **full-model** LLM BER/budget plan remains blocked. Separate, narrower 96-bit targeted studies are frozen below and must be labeled by their restricted fault support.

The bounded 10-request source script is `scripts/run_transformer_lm_smoke_once.sh`; completed runs verify and reuse finalized artifacts. It must not be treated as a production one-shot campaign. The frozen targeted 96-bit plans and foreground scripts are documented in [the prospective targeted protocol](TRANSFORMER_LM_TARGETED96_PROTOCOL.md). Native-BF16 Pythia computation, INT8/INT4 paths and gradient/Hessian baselines remain outside the validated scope.

## Mac export commands

```sh
scp kona:<repo>/workspace/implementation-worktrees/transformer-integration/workspace/runs/pythia160m_fp32stored_fp32compute_fault_smoke_v1/complete.zip ~/Downloads/pythia160m_fp32stored_fp32compute_complete.zip
scp kona:<repo>/workspace/implementation-worktrees/transformer-integration/workspace/runs/pythia160m_bf16stored_fp32compute_fault_smoke_v1/complete.zip ~/Downloads/pythia160m_bf16stored_fp32compute_complete.zip
scp kona:<repo>/workspace/implementation-worktrees/transformer-integration/workspace/runs/qwen3_0_6b_bf16_fault_smoke_v1/complete.zip ~/Downloads/qwen3_0_6b_bf16_complete.zip
```
