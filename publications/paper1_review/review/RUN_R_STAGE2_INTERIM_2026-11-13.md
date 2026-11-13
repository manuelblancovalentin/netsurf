# Run R, Stage 2 interim look at the main arms (13 November 2026)

At the researcher's request, after the Stage 2 main evaluation ended (576 jobs, no failures) and while the MC phase
runs (0 of 2,304 MC searches finished at 10:55). Descriptive, as recorded in the plan (interim-look note after
analysis note A2). Script `workflows/runR_stage2_analyze.py --interim-main` (branch `paper1-runR-v2-validated`,
6c8b185), committed and tested on synthetic data before any Stage 2 outcome was read. Output
`workspace/runs/paper1_runR_stage2_interim_main_v1/` (read-only), `summary.json` sha256 `695765786352d181…`.

Arms: abs64, abs1024, pos1024, squared1024, FKeras, significance-first (msb), the analytic rate-aware mask (v1) and
the selected mask (v2, R-A5). Not yet included: the MC-corrected mask, and so rule R2c. Nine checkpoints, four
budgets (6.25-25%), eight rates; 21-29 usable cells per checkpoint.

## Rules, interim

| Rule | Interim result | Can the confirmatory run change it? |
|---|---|---|
| R2a (v1 never resolved 0.5 point behind) | **fails**: 12 violations on 3 checkpoints (pinned 7, retrained 7101 2, retrained 7105 3), 0.60-1.17 points, all at 18.75-25% budgets and rates $2\times10^{-3}$ to $10^{-2}$, against abs1024, squared1024, pos1024, abs64 | Very unlikely: most violations have Holm p of 0.0015, far below 0.05 even with one more arm in the family |
| R2b (static rankings are not) | **holds**: abs1024 fails on 8 of 9 checkpoints, pos1024 on 9 of 9 | Only through the MC arm and the larger Holm family |
| R2a′ (v2 never resolved 0.5 point behind any static arm) | **holds**: no violation on any checkpoint | Barely: the MC arm is not in this rule; it can only change which cells are usable |

## Largest gap to the best arm, points (per checkpoint, over usable cells; biased upward)

| Arm | Range over the nine checkpoints |
|---|---|
| v2 (selected) | 0.11-0.61 |
| v1 (analytic) | 0.42-1.34 |
| abs1024 | 0.47-3.01 |
| squared1024 | 0.48-3.00 |
| abs64 (the paper's mask) | 1.17-3.04 |
| pos1024 | 5.0-56.8 |
| FKeras | 19.1-28.2 |
| significance-first | 23.5-52.1 |

v2 minus v1 per usable cell: median 0.00, worst $-0.22$, best $+1.34$ points. In usable cells v2 kept the v1 mask in
110-202 of the replication-cells per checkpoint, chose pos1024 in 27-71 and abs1024 in 3-20.

## Reading

- The static rankings each fail somewhere on ResNet-8. The absolute mask is behind v1 at low and middle rates (median
  over checkpoints and budgets 0.55 point at $3\times10^{-4}$, 0.79 at $10^{-3}$, 0.74 at $2\times10^{-3}$; up to 3.0
  points on the under-trained checkpoint at $2\times10^{-3}$); the positive mask is behind at high rates (median 1.0
  point at $5\times10^{-3}$, 8.0 at $10^{-2}$, 50 at $3\times10^{-2}$). That is the fault-rate dependence the paper is about, now on all nine
  checkpoints.
- The analytic rate-aware mask (v1) is usually as good as the best mask, but loses 0.6-1.2 points in a few cells near
  the switch on three of the nine checkpoints. That fits the overfitting found on training-side data.
- The selected mask (v2) removes those losses. It is never resolved half a point behind any static mask on any
  checkpoint, and its worst gap is the smallest of all arms on every checkpoint.
- Against the DS-CNN result (R4d′ failed narrowly, 0.50-0.68 points behind abs1024 in three cells), the claim that
  stands is per model: on ResNet-8 the selected mask is never far behind; on the DS-CNN it is close but not within
  half a point everywhere.

## Caveats

Interim; the confirmatory analysis runs once after the MC phase. v2 was designed after the v1 overfitting was seen on
training-side data from the same models (R-A5); its test windows and fault fields are untouched by that development.
The largest gap is measured against the best arm on the same fields, so it is biased upward for every arm.
