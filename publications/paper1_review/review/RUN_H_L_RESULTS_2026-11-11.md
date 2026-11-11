# Runs H and L: activation faults, checksums and correcting codes (exploratory, 11 November 2026)

Not used in the manuscript. DistilBERT W8A8, the study's frozen 512 sentences, clean accuracy 91.4%, two fault draws per point, no intervals. Designs are in the headers of `workflows/activation_abft_check.py` (Run H) and `workflows/correction_mechanisms.py` (Run L); both were committed before launch. Outputs: `workspace/runs/paper1_runH_activations_abft_v1/`, `workspace/runs/paper1_runL_correction_mechanisms_v1/`. Plots: `workspace/figures_for_discussion/`.

## Run H: weight faults against activation faults at the same per-bit probability

| Fault rate | 1e-4 | 3e-4 | 1e-3 | 3e-3 | 1e-2 |
|---|---|---|---|---|---|
| Activation faults only | 89.8 | 87.4 | 70.5 | 57.5 | 52.1 |
| Weight faults only | 90.7 | 90.9 | 90.6 | 86.7 | 61.5 |
| Weight faults, 12.5% absolute-gradient mask | 90.6 | 90.6 | 91.2 | 91.1 | 90.7 |
| Both, weights masked | 90.8 | 86.2 | 68.6 | 55.6 | 51.2 |

- Activation faults are redrawn at every forward call; about 4.2e7 activation bits are exposed per sentence against 3.4e8 stored weight bits.
- At equal per-bit probability, activation faults degrade the model about ten times sooner.
- A per-layer output checksum (stored column sums, fault-free) fails on 100% of sentences under weight faults at every rate from 1e-5, and on 0% under activation faults, because it reads the same corrupted activation as the multiplier.

## Run L: selective triplication against blanket correction (every stored bit, copies and check bits included, flips with p)

| Scheme | Extra storage | Highest p within 5 points | Accuracy at p = 0.01 | at 0.03 |
|---|---|---|---|---|
| No protection | 0% | 1e-3 | 59.1 | 55.7 |
| Row checksum correction (one faulty weight per row) | ~1% | 1e-3 | 59.1 | 55.7 |
| SEC-DED over 64 / 32 / 16 / 8 data bits | 12.5 / 21.9 / 37.5 / 62.5% | 3e-3 / 3e-3 / 1e-2 / 1e-2 | 76.2 / 83.2 / 87.4 / 91.0 | 52.9 / 54.1 / 60.6 / 54.2 |
| TMR on all sign bits (FORTUNE-style) | 25% | 3e-3 | 83.3 | 54.8 |
| TMR on the absolute-gradient mask, 6.25 / 12.5 / 18.75 / 25% of bits | 12.5 / 25 / 37.5 / 50% | 1e-2 / 1e-2 / 1e-2 / 3e-2 | 90.6 / 91.1 / 90.1 / 91.2 | 70.3 / 81.8 / 85.3 / 86.5 |

- At equal extra storage, triplicating the ranked mask tolerates about three times the fault rate of a SEC-DED code at 12.5% and 25% storage, and keeps 90-91% at p = 0.01 where the codes give 76-83%.
- At 25% storage, ranked TMR keeps 91.1% at p = 0.01 against 83.3% for TMR on all sign bits.
- The row checksum repairs 88% of faults at 1e-5 and almost none from 1e-3 on, because almost every row then holds two or more faulty weights.
- Below 1e-3 every scheme is within a point of clean; the codes leave far fewer faulty bits there (SEC-DED-8 leaves 1.2% of raw faults at 1e-3, ranked TMR 87.5%), so for very rare faults the codes are the efficient choice.

## Not modelled, all of which matter for a hardware claim

Logic area and power of voters, encoders and decoders; miscorrection of three or more errors (favours the codes); interleaving; scrubbing, which turns accumulated faults into faults per scrub interval and changes which p applies; a code combined with a mask. One model, one checkpoint, two draws.
