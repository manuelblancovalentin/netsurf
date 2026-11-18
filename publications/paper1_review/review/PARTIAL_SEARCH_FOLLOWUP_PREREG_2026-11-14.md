# Partial-group search follow-up: pre-registration (14 November 2026)

Written and committed before any run below exists. Requested by the researcher on 14 November 2026 ("Run them")
after reading the C1 results (`PARTIAL_GROUP_SEARCH_C1_RESULTS_2026-11-14.md`).

Replication 0 has already been inspected, so every test here uses **replications 1–5 only**.

## Questions

C1 found that the greedy slice search (4 slices per group, K = 32 fault fields per step) loses to abs1024 at fault
rate 1e-2, even on its own search clips. The suspected cause is selection noise: each step takes the best of about 80
noisy gains. Two follow-ups test this.

1. **More fault samples (K = 128).** Does the same greedy slice search, with 4 times more fault fields per step,
   (a) catch up with or beat abs1024, and (b) beat the K = 32 search? If the noise explanation is right, K = 128
   beats K = 32.
2. **Swap search from abs1024.** Starting from the abs1024 mask, can verified exchanges of bits between layer × bit
   groups beat abs1024? A swap is accepted only if it wins on fresh fault fields. This tests whether abs1024 is
   locally improvable at all once the winner's curse is guarded against.

## Fixed settings (all arms)

- **Model and data:** DS-CNN, 176,128 weight bits, INT8 stored-bit XOR faults.
- **Rate:** 1e-2 only. This is where C1's losses were largest.
- **Budgets:** 12.5%, 18.75%, 25% and 37.5% of the bits.
- **Search clips:** calibration clips 0–511 of the replication. The abs scores come from all 1,024 calibration clips,
  computed on CPU, and must rebuild the stored abs1024 masks exactly (already checked for replications 0–5:
  `paper1_partial_search_scores_v1`).
- **Evaluation:** the Stage 4 protocol, unchanged.
  - TFLite engine check, 800-clip test window of the replication, 8 fault fields with seeds [20261119, rep, field].
  - Paired by field with the stored Stage 4 masks (abs1024, msb, fkeras, ra_analytic, …).

## Arms

- **`partial_k32`:** the C1 search as it is (4 slices per group, K = 32, seeds [20261215, rep, rate index, round,
  draw]). Output `paper1_psearch_d_k32_v1`.
- **`partial_k128`:** identical except K = 128. The first 32 fields of each round are the K = 32 search's fields.
  Output `paper1_psearch_d_k128_v1`.
- **`swap`:** one independent search per budget, which keeps the bit count fixed. Output `paper1_psearch_d_swap_v1`.
  - **State:** within every layer × bit group, the protected bits are always a prefix of the group's bits in abs
    order (highest score first, ties by storage order). abs1024 is such a prefix in every group. The state is one
    protected count per group, starting at abs1024's counts.
  - **Move:** drop the last 256 protected bits of one group and protect the next 256 bits of a different group.
  - **Each round:**
    - **Propose.** On K = 32 fault fields (seeds [20261217, rep, rate index, budget index, round, 0, draw]), measure
      the paired accuracy of the current mask and of every one-group change:
      - each group that can grow by 256 bits, grown (gain);
      - each group with at least 256 protected bits, shrunk (loss).

      The proposed swap maximises gain(in) − loss(out) over pairs of different groups. Ties go to the smaller
      in-group id, then the smaller out-group id.
    - **Verify.** On 128 fresh fields (seeds [20261217, …, round, 1, draw]), compare the swapped mask with the
      current mask, paired by field. Accept only if the mean gain is positive and above 2 standard errors.
    - **Stop:** after 3 rejections in a row, or after 40 rounds.
  - With no accepted swap, the output equals abs1024.

## Resources (Quest, `gengpu` A100s; evaluation on `short` CPUs)

At about 16 ms per network pass:

| Arm | Jobs | Size of each job | Hours each | Total |
|---|---|---|---|---|
| K = 32 | 5 paths | about 190 rounds | about 1.8 | about 9 A100-hours |
| K = 128 | 5 paths | 4× the K = 32 passes | about 7.5 | about 37 A100-hours |
| Swap | 20 searches (5 reps × 4 budgets) | at most 40 rounds × about 5,400 passes | at most 1 | at most 20 A100-hours |

- **Total:** at most about 66 A100-hours.
- **Limits:**
  - K = 32: 4 h wall time, 120 s per round.
  - K = 128: 12 h wall time, 480 s per round.
  - Swap: 2 h wall time, 240 s per round.
- **Evaluation:** 8 CPU tasks per arm and replication, about 2 minutes each.

## Analysis (fixed now; `workflows/runR_psearch_followup_compare.py`)

For each arm, budget and replication: the mean over the 8 fields of (arm − comparator), in accuracy points. Then,
across the 5 replications: the mean, its standard error, a two-sided t test with 4 degrees of freedom, and a 90% t
interval.

- **Primary family:** arm − abs1024 for the 3 arms × 4 budgets = 12 tests, Holm at 0.05.
  - A cell is **ahead** or **behind** if its Holm-adjusted p < 0.05; otherwise it is **unresolved**.
  - It is **equivalent within ±1 point** if the 90% interval lies inside ±1.
- **Selection-noise family:** partial_k128 − partial_k32 for 4 budgets, Holm at 0.05.
- **Descriptive:**
  - each arm against msb, fkeras and ra_analytic;
  - bits differing from abs1024;
  - for the swap arm: accepted and rejected swaps, and which groups gained or lost bits;
  - for the greedy arms: the round where the accepted gain falls below its own standard error.

## How the outcomes will be read (fixed now)

- **K = 128 beats K = 32 in at least one budget, and none goes the other way:** this supports the selection-noise
  explanation.
  - If K = 128 also ties or beats abs1024, the greedy search's C1 losses were a noise artefact.
  - If it is still behind abs1024, noise is part of the cause but not all of it.
- **K = 128 does not beat K = 32 in any budget:** this is evidence against noise as the main cause. Greedy short-sightedness,
  or abs1024 being near a local optimum, become the leading explanations.
- **Swap ahead of abs1024 in some budget:** abs1024 is locally improvable at 1e-2. Report by how much and which
  groups moved.
- **Swap unresolved or within ±1 everywhere, or few swaps accepted:** abs1024 is locally hard to improve at 1e-2 by
  exchanges of 256 bits between groups. This supports the paper's statement that abs1024 holds up at this rate.
- **Scope:** none of these results changes the C1 or Phase A conclusions on replication 0. All are one model and one
  rate.
