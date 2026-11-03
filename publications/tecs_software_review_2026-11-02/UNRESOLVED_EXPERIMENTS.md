# Explicit unresolved-experiment ledger

This is a manuscript planning ledger, not a set of executed results. Frozen
source studies remain untouched. The current draft's main INT8 curves and
published-form FKeras comparison are unchanged.

| Priority | Missing scientific discrimination | Current evidence | Required next test / decision |
|---|---|---|---|
| 1 | Does exact isolated ranking outperform a representation-aware exponent-first baseline on targeted FP32/BF16 bits? | Three finalized 96-bit LM pilots show sign-versus-exponent mask allocation; Pythia penalties dominate some means. | Freeze an exponent-first policy, same 96 bits, budgets, 256-token cohort and paired fields; compare finite NLL, nonfinite failures and capped all-token NLL. No whole-model claim. |
| 2 | Why did the DistilBERT capped-loss ordering reverse? | Original study favors significance; independent validation-cohort replication favors exact isolated weakly, with tied accuracy. | Factorially vary calibration cohort, evaluation cohort and fault fields one at a time under a new protocol. Preserve the two old studies separately. Add both classifier output rows and earlier layers only in a new architecture-stratified protocol. |
| 3 | Do targeted LM effects transfer beyond the chosen 32 words and short cohort? | Each LM pilot protects at most 24 of 96 sampled bits; 256 final target tokens and one model checkpoint. | Prespecify a broader, stratified storage population and an independent longer held-out test; measure cost before approval. Distinguish BF16-stored/FP32-compute from native BF16. |
| 4 | Is any conditional-allocation advantage over exact isolated ranking robust? | Independent-pool intervals all include zero; small effects and costly response tables. | Await separately frozen equal-compute search and M11 studies. Do not report ongoing outcomes or infer that all static rankings fail. |
| 5 | Can the common INT8 grid separate published FKeras, corrected MSB and gradients on the same large cohort? | Verified 128-clip partial-tier study and separate 1000-clip main curves; neither may be pooled. | Run the frozen published-only FKeras v2 plan after researcher preflight approval. Do not substitute the unrun v1 repository-form score. |
| 6 | Does ToyCar AUC ordering persist with more independent fields and inputs? | Eight fields on previously exposed complete files. | More fields improve fault-field precision only. Fresh balanced files are needed for input-population confirmation. |
| 7 | Is the paper sufficiently novel for TECS beyond FCCM/FKeras/PrioriFI and selective-ECC literature? | Controlled exact-bit evaluation and task diversity are real; no universal new optimizer or protection principle is established. | Obtain author-level novelty/extension decision after the prespecified missing comparisons. Keep the present draft internal until resolved. |

No new full-model LLM, physical hardware, radiation-calibrated reliability,
equal-compute-search, or M11 scaling result is inferred from these pilots.
