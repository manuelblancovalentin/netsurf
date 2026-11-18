# Paper 1: briefing for the advisor meeting (11 November 2026)

Includes the results of the overnight runs. Draft: 21 pages, `delivery_v8_tecs_revision`.

## In one breath

The rejected TCAD paper sold a new metric and a hardware flow it could not back up. The new TECS draft is a careful software-only measurement: which stored bits are worth protecting at a fixed budget, on six frozen INT8 models, judged by task quality. Its main recommendation, the per-example absolute-gradient ranking as a default, was confirmed on ResNet-8 with rebuilt masks on new data. On DS-CNN a second, larger run confirmed it too, on test inputs used before.

## 1. What was wrong with the TCAD paper

- It claimed qPolarGrad finds vulnerable bits better, without rigorous evidence. A round-one reviewer asked exactly this.
- Its gains (up to +21.7 points over MSB-first) were quoted at the budget where the lead was widest.
- Three small 6-bit models, one checkpoint each, loose statistics.
- FKeras was described as a diagonal-Hessian method; it is not.
- The "much cheaper than Hessian" argument does not hold: Hessian scoring takes seconds in the current code.
- The TMR and resource numbers rested on a cost model, not on measured hardware.

## 2. What the new paper is

- **Scope:** software-only evaluation. No new metric, no hardware, no radiation claim. qPolarGrad is gone.
- **Question:** with a fixed protection budget and many simultaneous stored-bit faults, which bits are worth protecting?
- **Evidence:** six frozen INT8 models (ResNet-8, DS-CNN, ToyCar, DistilBERT, Pythia-160M, Qwen3-0.6B; up to 4.8 billion bits), seven rankings including published-form FKeras, eight BERs by four budgets, every comparison paired on the same fault fields.
- **Discipline:** a usefulness criterion fixed before comparing (accuracy within 5 points of clean, or loss within 0.5 nats per token). We only draw conclusions where at least one ranking keeps the model useful.

## 3. What we claim, with numbers

1. **Ranking choice matters where the model is still useful.** At 12.5% protection the best ranking tolerates about 3 times the BER of MSB-first on four of the six models, and 3 to 30 times that of no protection. On ResNet-8 at BER 0.003 the absolute-gradient mask loses 3 accuracy points, MSB-first 16, no protection 44.
2. **Per-example absolute gradient is a strong default among the rankings we measured.** In useful cells it is within 0.2 accuracy points of the best ranking on the three classifiers and within 0.16 nats per token on the two language models. It is never resolved worse than another ranking on the classifiers.
3. **Protection can hurt.** The positive-gradient ranking is worse than no protection in part of two transformer grids (up to 6 accuracy points on DistilBERT), because the faults it leaves push the weights in one direction.
4. **Theory, as background only:** no static ranking is guaranteed optimal across BER and budget. We give two small counterexamples and say this is standard reliability theory.

## 4. What the overnight runs showed

**Run A, confirmation on new data (ResNet-8 and DS-CNN).** Everything except the trained checkpoint was redrawn: new calibration inputs, rebuilt masks, new test windows, new fault fields. The pass rule was fixed before the run: regret of the default below 1 point, with 95% confidence, in every useful cell.

- **ResNet-8 passed.** Nine new windows. Largest regret 0.19 points, upper bound 0.64, 63 resolved wins and no loss.
- **DS-CNN did not pass the first time, and passed the second.** With three windows the regret could not be bounded below 1 point. A second run with eleven replications bounds it below 0.2 points, with 48 resolved wins and no loss. Caveat: DS-CNN has only 4,890 test clips, so the second run reuses test inputs; its calibration draws, masks and fault fields are new.
- **DS-CNN margin over MSB-first is modest.** MSB-first loses at most 3 points there.
- **One headline number went down.** The earlier draft said the default tolerates 10 times the BER of MSB-first and 30 times that of no protection on ResNet-8. On the nine new windows it is 3 and 10. The original window sat just outside the 5-point margin at two grid points. The accuracy gaps held (13.1 points over MSB-first at BER 0.003, against 12.7 before).
- **FKeras control.** Inside a half-protected bit tier, FKeras's Hessian order beats a random fill (by up to 16 points on ResNet-8). It is not better than plain storage order, and is worse by 4.0 and 2.5 points in two cells outside the useful region.

**Run B, loss cap on the language models.** We replayed every saved fault field of Pythia and Qwen to record how often token losses hit the 30-nat cap.

- All 9,600 outcomes reproduced bit for bit.
- In the useful region the cap is irrelevant (one capped token on Pythia, none on Qwen) and changes no comparison.
- Outside it the cap dominates on Pythia (83% of tokens at BER 0.03), so those loss values are lower bounds. The "protection can hurt" result keeps its sign without the cap.

## 5. What we do not claim

- Hardware cost, TMR, area, power, or a radiation rate.
- That a universal ranking has been disproved empirically, or that absolute gradient is optimal.
- That MC-guided or conditional selection helps: the one test we have was too small to tell either way.
- Anything across trained checkpoints (one per model).
- Confirmation on ToyCar or the transformers; confirmation on untouched DS-CNN inputs.
- Superiority over learned or search-based allocators (not measured).

## 6. Where a reviewer will push, and what would answer it

| Weakness | What would answer it |
|---|---|
| One trained checkpoint per model | Retrain three to five ResNet-8 and DS-CNN checkpoints and repeat Run A. Needs approval and compute. |
| Transformers not repeated | A Run A for DistilBERT. Several nights of CPU. |
| No hardware cost; gradient masks are irregular | A structured variant (per tier or per layer) or a metadata-cost proxy. New work. |
| No learned or search-based allocator | Out of scope, stated as a limit. |
| Dense-error regime (BER 1e-4 and above) | Stated openly; it is where the masks differ. |

## 7. Decisions to ask for

1. Submit to TECS now as a measurement paper with honest limits, or first add retrained checkpoints and a larger DS-CNN confirmation?
2. Approve retraining of the two small classifiers?
3. Does the side investigation below go into this paper later, or into a follow-up?

## 8. Side investigation, not in the paper (exploratory)

This is the cleaned-up version of the "polar" idea.

- **Idea.** For any mask and BER, each weight has an exact average shift (drift) and an exact variance under the faults left unprotected. Both follow from the stored bits in closed form.
- **Test.** Replace the fault field on DistilBERT by Gaussian noise with exactly those two moments per weight and run a few forward passes, with no bit-level fault sampling.
- **Result.** It predicts the measured accuracy of every mask and BER with a mean error of 1.3 points (78 conditions). Where masks differ by at least 2 points, choosing the mask with this model costs 0.03 points on average and ranks the masks almost perfectly (rank correlation 0.94). Drift alone or variance alone do much worse (6.2 and 4.2 points).
- **What it explains.** The positive-gradient harm is pure drift. The collapse of MSB-first at high BER needs both terms: protecting only the sign bit pushes small weights away from zero.
- **Where it fails.** For unprotected and random masks at intermediate BER it is too optimistic by 12 to 17 points, because rare sign-bit flips are not Gaussian.
- **Why it matters.** If two numbers per weight are enough, mask design becomes an optimization of those two numbers at the target BER, which is cheap and uses exactly the cancellation and interaction the original idea was after.
- **Status.** One model, one cohort, a prototype. Building a mask from it and testing it on the frozen fault fields is the next step and needs approval.
