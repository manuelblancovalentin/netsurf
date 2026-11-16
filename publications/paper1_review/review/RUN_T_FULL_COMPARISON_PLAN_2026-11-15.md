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
