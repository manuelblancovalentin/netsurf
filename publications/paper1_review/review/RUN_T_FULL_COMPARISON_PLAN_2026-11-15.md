# Run T: one comparison for every model, with real protection schemes (plan, 15 November 2026)

**Status:** design written at the researcher's instruction ("Then we need those runs!"). No Run T job has been
launched. Each stage gets a timing smoke and a frozen analysis before its own launch.

## 1. The requirement, in the researcher's words

All results should contain:
- non-ideal protection (a protected bit can still flip);
- the ECC / ABFT / TMR hardware comparison;
- FKeras, MSB-first (FORTUNE) and our rankers, and any other the literature poses as relevant;
- all models, from tiny to huge.

## 2. What already exists (two corrections to my earlier summary)

- **ECC, ABFT and TMR with non-ideal protection were already run on DistilBERT** (Run L, 11 November, exploratory,
  2 fault draws) as well as on ResNet-8 (Run L2, 8 draws, in the paper). I wrongly said "never run on anything but
  ResNet-8".
- **FKeras on the language models is half-determined already.** FKeras orders by bit position first, then by the
  Hessian score. At 12.5% and 25% of bits it protects exactly the same bits as MSB-first on every model, whatever the
  Hessian says. Those cells are measured. The Hessian only matters at the half-filled tiers (6.25% and 18.75%).
  Those are the cells that are missing.

| Model | abs-gradient masks | FKeras Hessian | ECC / ABFT / TMR, non-ideal |
|---|---|---|---|
| ResNet-8 | yes | yes | yes (Run L2, 8 draws, one window) |
| DS-CNN | yes | yes | **missing** |
| ToyCar | yes | yes | **missing** |
| VWW, jet MLP, MNIST MLP | yes (1,024 inputs) | **missing** | **missing** |
| DistilBERT | yes | **missing** | partly (Run L, 2 draws, no FKeras/MSB partial tiers) |
| Pythia-160M, Qwen3-0.6B | yes | **missing** | **missing** |

## 3. One protocol for every model

**Fault model:** every stored bit flips independently with probability p. This includes the extra copies of
triplicated bits and the check bits of codes. Nothing is immune.

**Conditions**, all on the same raw fault field within a draw:

| Condition | How it fails | Extra storage |
|---|---|---|
| Unprotected | every flip applies | 0 |
| TMR of ranked bits, for each ranker | a voted bit is wrong with probability 3p² − 2p³ | 2 × the share of bits protected |
| SEC-DED over 64 / 32 / 16 / 8 data bits | one error per word repaired; two or more left as they are; check bits also flip | 12.5 / 21.9 / 37.5 / 62.5% |
| ABFT checksum correction | repairs one faulty weight per output channel (or matrix row); more are left | about 1% |
| ABFT with faulty checksums and wrong repairs | as above, with checksum words also flipping (sensitivity check) | about 1% |

- **Rankers for TMR, main text:** per-example absolute gradient (our best static), FKeras, MSB-first. MSB-first at
  12.5% of bits is FORTUNE-style sign-bit triplication.
- **Rankers for TMR, appendix:** positive gradient, global gradient, numerical jump, random.
- **Shares of bits protected:** 6.25, 12.5, 18.75 and 25% (extra storage 12.5, 25, 37.5 and 50%).
- **Fault rates:** the study grid, 1e-5 to 1e-1 (8 rates).
- **Statistics:** paired by fault draw; the usefulness margin and the Holm-corrected sign-flip tests of the paper.

This puts everything the researcher listed in one figure per model: accuracy (or loss) against fault rate at equal
extra storage, with our ranker, FKeras, FORTUNE, ECC and ABFT as the lines.

## 4. Models

| Size | Models |
|---|---|
| Tiny | jet MLP (epoch 100), MNIST MLP (epoch 30) |
| MLPerf Tiny | DS-CNN, ResNet-8 (pinned, and under-trained), VWW, ToyCar |
| Large | DistilBERT, Pythia-160M, Qwen3-0.6B |

The other jet, MNIST and ResNet-8 checkpoints stay in the ranking-choice studies, which already cover them.

## 5. Stages, code and cost

About 34 conditions × 8 rates = 272 evaluations per fault draw. Times per evaluation below are from earlier runs;
**every estimate is rough until that stage's timing smoke.** All stages run on kona's CPUs (80 threads, 471 GB of
memory); no GPU is needed.

| Stage | What is new | Draws | Rough cost |
|---|---|---|---|
| **T1: TFLite models** (jet, MNIST, DS-CNN, ResNet-8 ×2, VWW, ToyCar) | generalise the Run L2 runner to any INT8 TFLite model; FKeras Hessian for VWW, jet and MNIST through the generic companion | 24 (6 windows × 4); ToyCar 8 | 100–150 core-hours |
| **T2: DistilBERT** | FKeras Hessian for a transformer (Lanczos with Hessian-vector products, checked against an exact Hessian on a tiny model); rerun Run L with FKeras and MSB-first at the half-filled tiers | 8 | about 100 core-hours |
| **T3: Pythia and Qwen** | the same Hessian code; the Run L runner adapted to their executables | 8 | 100–300 core-hours; the Qwen Hessian needs 60–100 GB of memory |

Total: roughly 300–550 core-hours, one to two days of kona if nothing else runs.

**Order:** T1 first (smallest new code, covers six models), then T2, then T3.

## 6. What this changes in the paper

- The main comparison becomes Run T: non-ideal protection, every model, the five families of lines.
- The current ideal-protection grid moves to the appendix as the "perfect protection" limit.
- The paper keeps the caveat that only storage is counted: voters, encoders, decoders and scrubbing are not.

## 7. Defaults the researcher can change

1. Main-text figure at 12.5% and 25% extra storage (TMR of 6.25% and 12.5% of bits). At 25%, FKeras and MSB-first
   are the same mask; at 12.5% they differ.
2. Jet and MNIST in the main text: the fully trained checkpoints.
3. ResNet-8: pinned and under-trained.
4. Draws: 24 for the TFLite models, 8 for ToyCar and the transformers.
5. Pythia is included (three language models, not two).

## Amendment A1 (15 November 2026, the researcher's instructions; nothing launched yet)

1. **As many models as we can.** Run T covers every checkpoint we hold, 24 in all: jet MLP (5 epochs), MNIST MLP
   (4 epochs), ResNet-8 (9 checkpoints), DS-CNN, VWW, ToyCar, DistilBERT, Pythia-160M and Qwen3-0.6B. This replaces
   defaults 2 and 3 of Section 7.
2. **Use Quest to cut the wall time.** What a GPU can and cannot do here:
   - **TFLite models:** the faulted runs use the TFLite reference kernels, which run on CPUs only. Quest helps by
     running hundreds of tasks at once (CPU job arrays), not by GPU. About 250–300 core-hours, one to two hours of
     wall time once the jobs start.
   - **Language models:** the W8A8 executable multiplies INT8 matrices with a CPU-only operator
     (`torch._int_mm`, `kona_assets/int8_torch.py`). A GPU run needs a second, exact path (64-bit floating-point
     products, which are exact for sums of this size) that must first reproduce the CPU executable bit for bit on
     clean and faulted checks. The models, masks and sentences must also be copied to Quest.
   - **Hessians for FKeras:** well suited to a GPU.
   So: T1 on Quest CPU arrays; T2 and T3 on Quest GPUs after the exact-path check passes. If the check fails on a
   model, that model runs on kona's CPUs.
3. **FKeras, as published and as coded.** The FKeras paper (Section 3.1; our hashed copy, sha256 `bb327b55…`) ranks
   weights by the Hessian score, then orders bits tier by tier: "we rank all of the MSB's by the weight sensitivity
   score, then the second MSB's, and so on". It also says this "is not ideal" and that mixing bits of different
   weights is future work. The FKeras source has a second function, `rank_bits`, which scores a bit by the weight's
   score times its power of two and so does mix tiers. Run T adds that as a separate arm, `fkeras_weighted`, if the
   function is present at the pinned commit (to be checked; the local copy examined is not the pinned file). The
   published tier-first order stays the arm named FKeras.
4. **The ideal-protection grid stays in the draft for now** and is removed once the non-ideal results exist for
   every model. The conclusions are re-derived from the non-ideal results, not assumed.

## Note A2 (15 November 2026): sources checked for the FKeras and PrioriFI text

- FKeras paper, our hashed copy (sha256 `bb327b55…`), Section 4.2.2: the tier-first order "is not ideal, and a better
  bit-level ranking can be achieved by mixing bits of different significance … We leave that as future work."
- FKeras source at the pinned commit 68e775c (`fkeras/metrics/hessian.py`, sha256 `3e5b61b8…`, matching the hash on
  record): `rank_bits` (weight score × 2^bit, tiers mixed) is defined; the ranking path calls `sort_bits_MSB_to_LSB`
  (tier first). Run T labels the mixed arm "fkeras_weighted: defined in the FKeras source, not used by it".
- PrioriFI, author PDF (ASPLOS 2026; sha256 `8cce24b6c8cce8d2…`), Section 3.1: "The single-bit
  flip model, which simulates scenarios where only one bit flips at a time"; its goal is to find sensitive bits with
  fewer injections; it builds no protection mask. An earlier NetSuRF adaptation (worktree `m12-priorifi-crossbench`,
  `docs/PRIORIFI_CROSSBENCH_PREFLIGHT.md`) reached the same reading from its public code.
