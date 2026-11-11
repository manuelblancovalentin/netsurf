# Empirical evidence handoff

All tables/figures in this directory are derived from finalized source measurements. Every plotted condition mean was reconstructed from saved policy-field observations; no inference was run. Hash receipts record the exact data used. Large original mask/event/score payloads are linked by their source manifests rather than all rehashed in this manuscript build.

## Main current comparisons

| Model | Eligible stored INT8 bits | Final cohort | BERs | Fields / BER | Methods | Conditions | Policy-field observations |
|---|---:|---|---:|---:|---:|---:|---:|
| ResNet-8 / CIFAR-10 | 618,880 | 1000 examples | 8 | 32 | 7 | 240 | 7680 |
| DS-CNN / Speech Commands | 176,128 | 1000 examples | 8 | 32 | 7 | 240 | 7680 |
| Autoencoder / ToyCar | 2,113,536 | 64 complete files | 5 | 32 | 7 | 150 | 4800 |
| DistilBERT / SST-2 | 344,469,504 | 512 examples | 8 | 24 | 6 | 208 | 4992 |
| Pythia-160M / WikiText-2 | 988,545,024 | 4096 scored tokens | 8 | 24 | 6 | 208 | 4992 |
| Qwen3-0.6B / WikiText-2 | 4,767,875,072 | 4096 scored tokens | 8 | 24 | 6 | 208 | 4992 |

All use 6.25%, 12.5%, 18.75%, and 25% interior protection, plus zero/full controls. ResNet, KWS, DistilBERT, Pythia, and Qwen BERs are 1e-5, 1e-4, 3e-4, .001, .003, .01, .03, and .1. ToyCar BERs are 1e-5, 1e-4, .001, .003, and .01.

The combined presentation has 1,254 conditions and 35,136 labeled policy-field observations over 1,248 separately scheduled fields across six studies. These totals are accounting, not independent cross-model replications. TFLite FKeras/MSB equality at 12.5% and 25% contributes 1,344 duplicate labeled outcomes; repeated full-protection controls are also not independent clean measurements. W8A8 full controls are reused clean references: 4,800 actual faulted requests/model versus 4,992 saved observations/model.

The three TFLite studies use seven methods and four schedule replications with eight fields each. Saved mask hashes verify four random masks and exactly one mask for every other method at each budget. The transformer studies use six methods and one sealed mask per method/budget. Confidence intervals condition on frozen model, cohort and masks; they do not quantify generalization over checkpoints or fresh inputs. For the TFLite random baseline, pooled field intervals combine four fixed sampled masks and do not separately identify population-level mask-construction uncertainty.

## FKeras and computation

Published formula only; absolute value of signed leading-Hessian projection, strictly within descending bit tiers. All three current TFLite studies use two signed largest-magnitude eigenpairs from a float64 continuous decoded-weight companion, ARPACK, tolerance 1e-5, and 25 HVPs per replicate. Activation requantization is removed; fixed decoded bias is excluded from differentiated/faultable population. No full integer-program Hessian is claimed. The same 64 ranking images/clips as gradients are used for ResNet/KWS; ToyCar uses all 3,136 windows from 16 training files, and evaluates protection on complete-file 1-AUC over 64 files/12,544 windows.

Score costs (mean of 4 builds, seconds):

| Model | Global gradient | Shared absolute/positive gradient | Rank-2 FKeras | Full study elapsed h | Coordinator peak GiB |
|---|---:|---:|---:|---:|---:|
| ResNet-8 / CIFAR-10 | 1.186 | 16.986 | 5.580 | 14.07 | 7.16 |
| DS-CNN / Speech Commands | 0.412 | 7.467 | 1.432 | 6.31 | 2.41 |
| Autoencoder / ToyCar | 3.848 | 208.426 | 11.421 | 3.24 | 18.19 |

Maximum normalized eigenpair residual across the four builds: ResNet-8 / CIFAR-10 6.07e-16; DS-CNN / Speech Commands 4.89e-16; Autoencoder / ToyCar 5.06e-16. All two leading eigenpairs converged. Exact eigenvalues, orthogonality errors, implementation versions, and HVP counts are retained in data/fkeras_numerical_receipts.csv.

Absolute/positive share a Jacobian; their timing must not be added. The rank-2 FKeras cost is not a claim about full Hessian analysis. Timestamp fields in older parent manifests reflect an early coordinator phase; use the explicitly saved coordinator elapsed receipt.

Measured wide-run request service times (sum over concurrent shards, not wall time):
- DistilBERT / SST-2: median 23.75 s, p95 27.79 s, summed service 30.88 h over 4800 requests. Peak memory and total score-building wall time are not recorded in these finalized wide-run manifests; no invented estimate is supplied.
- Pythia-160M / WikiText-2: median 11.56 s, p95 13.06 s, summed service 15.68 h over 4800 requests. Peak memory and total score-building wall time are not recorded in these finalized wide-run manifests; no invented estimate is supplied.
- Qwen3-0.6B / WikiText-2: median 32.55 s, p95 51.20 s, summed service 49.35 h over 4800 requests. Peak memory and total score-building wall time are not recorded in these finalized wide-run manifests; no invented estimate is supplied.

## Manuscript-critical findings

At 1% BER and 6.25% protection, FKeras−MSB is +3.0347 [1.8351, 4.2342] nats/image on ResNet but −1.6777 [−2.4252, −.9302] nats/clip on KWS; both original Holm-adjusted p=.01120. This across-model contrast is not pooled in common units. The ResNet 320-field focused comparison independently gives +2.2414 [1.8896, 2.5931], Holm p=.0012. The focused ToyCar 384-field comparison at 18.75% and 1% BER gives FKeras−absolute-gradient risk +.11059 [.09200, .12918], Holm p=.0012. Focused studies are supplemental, never pooled into the main grid.

At 1% BER, Pythia global-gradient−MSB is −4.8228 nats/token at 12.5% but +.7712 at 25%; both original 160-comparison Holm p=.016. The Qwen independent-field exploratory follow-up at 12.5% reverses absolute-gradient−MSB from −9.2968 [−10.0205, −8.5671] at 5% BER to +1.6368 [.6051, 2.6609] at 7.5%. Finite uncapped means reverse too; finite cap frequency nonetheless rises to 7.07% MSB/10.49% absolute at 7.5%.

The new KWS 32-pool prespecified primary conditional−exact-isolated effect is −.0000824 [−.0016935, .0015287] nats/clip, p=.9176. In 27/32 pools masks differ; decision changes did not establish utility improvement. Both conditional and exact-isolated improve slightly over the original absolute-gradient global context. The recovered campaign-level plan controls the primary comparison; generic per-pool report primary labeling is inconsistent and is not used.

These support condition-dependent effectiveness, not causal attribution to multibit interactions, not universal optimizer superiority, and not calibrated radiation reliability.

## Exact-bit score validation

data/empirical_validation.csv contains all six methods against 320 stratified exact candidate-bit damages, source bootstrap CIs and timing. These are sample-limited and are not population-weighted whole-network correlations. ToyCar positive-MSE versus positive-AUC correlations are kept separate in data/toycar_auc_correlations.csv. All 320 score identities were joined to their exact bit-damage measurements after streaming/hash-verifying the original prediction file.

## Missing/limited support

FKeras not measured on the three W8A8 transformers; PrioriFI-derived allocation has small engineering receipts but no finalized matched protection-effectiveness campaign located. Historical qPolarGrad absent from current main grids. Do not fill missing curves or compare a scheduler as if it were a validated protection allocator.

## Figure use

- reliability_ber_125.pdf: six model panels at 12.5%; reliability_ber_250.pdf: the same at 25%. The three classification panels use an explicitly labeled asinh axis (linear near 0) to preserve beneficial and catastrophic tails.
- reliability_budget_001.pdf: all six methods/seven where available, all four interior budgets, common 1% BER.
- paired_reversals_and_kws.pdf: interpretable paired effect signs and the full 32-pool distribution with mean CI.
- reliability_all_budgets_supplement.pdf: all 24 model/budget panels with every measured BER.
- SVG and PNG versions are supplied for review. All curves connect observed cells only. No inferred FKeras or PrioriFI results.
