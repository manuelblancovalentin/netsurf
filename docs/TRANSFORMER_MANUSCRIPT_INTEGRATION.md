# ACM TECS software manuscript: transformer evidence status

This updates the existing November 9 software-only manuscript evidence track. The three validated TinyML benchmarks remain its backbone. No historical manuscript artifact or finalized experiment is edited here.

| Target configuration | Actual storage → computation | Current NetSuRF evidence | Manuscript status |
|---|---|---|---|
| DistilBERT/SST-2 | FP32 → FP32 | Finalized 1,024-classifier-bit v4 campaign, 96 paired fields; [read-only audit](DISTILBERT_V4_RETROSPECTIVE_AUDIT.md) | Eligible as a narrowly scoped targeted-classifier result, with cap and calibration limitations disclosed |
| Pythia-160M/WikiText-2 | FP32 → FP32 | Finalized 32-token engineering fault smoke | No BER/budget effectiveness claim |
| Pythia-160M/WikiText-2 | BF16 → distinct FP32 execution copy | Finalized 32-token engineering fault smoke; encoded BF16 XOR precedes exact widening | No BER/budget effectiveness claim; native BF16 compute remains validation failed |
| Qwen3-0.6B Base/WikiText-2 | BF16 → BF16 | Finalized 32-token engineering fault smoke | No BER/budget effectiveness claim |

The authoritative readiness handoff used by the LM smokes is `NETSURF_ASSET_READINESS.json` SHA-256 `3192d3ce57b7a11089af4c1e0f802e851ebc8b179fd823a793fecf333313db35`. Each immutable smoke bundle includes that exact manifest, the reviewed NetSuRF runtime registry, source files, candidate identities, schedules, fault counts, task risks and restoration evidence. The three complete ZIPs were independently checked for manifest checksums and ZIP CRC integrity. Their exact receipts and limits are in [LM engineering evidence](TRANSFORMER_LM_ENGINEERING_EVIDENCE.md).

## Method text for the current draft

For a verified transformer, a fault coordinate indexes an actual stored FP32 or BF16 word of the declared CPU `nn.Linear.weight` population. NetSuRF XORs the encoded stored bit, checks the inference-consumed representation, and restores and hashes the original bytes after the trial. Embeddings, layer norms, biases, tied weights, and other operators are outside that population; eligible counts are per pinned model. For Pythia BF16 storage with FP32 computation, the BF16 word is mutated first and a distinct FP32 copy is rematerialized for inference. This is a different fault contract from native BF16 computation. A protection mask suppresses specified logical events under the same paired fault field. SST-2 uses public-label mean cross-entropy and accuracy. WikiText-2 uses deterministic teacher-forced target-token negative log-likelihood and perplexity, with each target scored once under the pinned 1,024-token context and 512-token stride. Nonfinite or failed target evaluations receive the prespecified 30-nat penalty and are counted separately.

The DistilBERT v4 result concerns only 32 classifier words. Its high-BER cross-entropy contrast is dominated by a few catastrophic, finite-logit fields under the prespecified cap. The new validation-cohort replication plan is frozen but not run. No current transformer evidence supports a general full-model reliability claim, physical upset-rate claim, or effectiveness curve for Pythia or Qwen. Accuracy, ROC-AUC and perplexity should remain on separate axes. Production cost and fault-exposure feasibility must be measured before adding LLM BER or protection-budget panels.
