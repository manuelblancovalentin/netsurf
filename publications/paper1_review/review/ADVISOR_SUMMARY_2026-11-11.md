# Paper 1: what changed since the TCAD submission (talking points, 11 November 2026)

## The old paper (TCAD, rejected)

- **Pitch:** NetSuRF as an end-to-end hardening flow built on a new metric, qPolarGrad, with an HLS/TMR cost model.
- **Headline claims:** 90–95% of baseline accuracy at 30–40% of full-TMR resources; up to +21.7 points over MSB-first and +14.2 over a Hessian ranker; far cheaper than Hessian methods.
- **What was wrong:**
  - No rigorous evidence that qPolarGrad finds the vulnerable bits better (a round-one reviewer asked exactly this).
  - Gains were quoted at the budget where the lead was widest.
  - Three small 6-bit models, one checkpoint each, loose statistics.
  - FKeras was described as a diagonal-Hessian method; it is not.
  - The runtime argument does not hold: Hessian scoring takes seconds in the current implementation.

## The new paper (TECS draft)

- **Scope:** software-only evaluation. No new metric, no hardware, no radiation claims. qPolarGrad is gone.
- **Question:** with a fixed protection budget and many simultaneous stored-bit faults, which bits are worth protecting, judged by task quality?
- **Evidence:** six frozen INT8 models (ResNet-8, DS-CNN, ToyCar, DistilBERT, Pythia-160M, Qwen3-0.6B, up to 4.8 billion bits), seven rankings including published-form FKeras, eight BERs by four budgets, every comparison paired on the same fault fields.

## What we claim now

1. **Ranking choice matters where the model is still useful.** At 12.5% protection the best ranking tolerates up to 10 times the BER of MSB-first and 30 times that of no protection (ResNet-8).
2. **Per-example absolute gradient is a strong default among the rankings we measured.** Within 0.2 accuracy points of the best on the three classifiers, within 0.16 nats/token on the language models.
3. **Protection can hurt.** The positive-gradient ranking is worse than no protection in part of two transformer grids, because it leaves a one-sided perturbation (mechanism checked on DistilBERT only).
4. **Theory as background:** no static ranking is guaranteed optimal across BER and budget.

## What we do not claim

- Hardware cost, TMR, area, power, or a radiation rate.
- That a universal ranking has been disproved empirically, or that absolute gradient is optimal.
- That MC-guided or conditional selection helps: the one test we have was too small to tell either way.
- Anything across trained checkpoints (one per model).
- Superiority over learned or search-based allocators (not measured).

## Running now

- **Run A:** repeats the classifier comparison on new inputs with rebuilt masks (ResNet-8 on ten windows, DS-CNN on four), plus a control that shows whether FKeras's partial-tier differences from MSB-first are real or a storage-order artefact. This is what turns claim 2 from exploratory into confirmed for these checkpoints.
- **Run B:** replays the two language-model grids to record how often token losses hit the 30-nat cap.
- **Prototype:** a first check of a drift-aware mask objective, the cleaned-up version of the "polar" idea.

## Honest status

A complete draft exists (19 pages). It is a major-revision-grade TECS submission until Run A is in. The main remaining weaknesses are one checkpoint per model and no hardware cost.
