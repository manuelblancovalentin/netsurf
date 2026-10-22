# M2 finalized-result diagnostic

This note analyzes the immutable finalized bundle
`workspace/runs/mnist_quantized_ranker_comparison/20261022T215214.228477Z-8306c57a`.
It does not alter or supersede that bundle. Its scientific payload SHA-256 is
`f0fa9c19929b577fb8d47f2d479a8730b60606182bc564f00d3a07bf2ce8a32a`.

## Directly recomputed paired results

The 12 fault trials were regrouped directly from `measurements/trials.csv` by
`trial` and `schedule_sha256`. The reported policy means are reproduced. The
first-order-gradient-times-exact-delta mask exceeded the then-current MSB mask
in accuracy in all 12 paired trials. This is evidence for those exact masks,
samples, schedules, and the 10% budget; it is not evidence against every
MSB-primary policy.

| Policy | Mean accuracy | 95% trial CI | Mean cross-entropy | Paired accuracy difference vs none |
|---|---:|---:|---:|---:|
| No protection | 0.731771 | [0.636629, 0.826913] | 1.213694 | — |
| M2 MSB-first | 0.784505 | [0.683835, 0.885176] | 0.994470 | 0.052734 [0.033495, 0.071974] |
| M2 random | 0.758464 | [0.657688, 0.859240] | 0.887621 | 0.026693 [-0.000306, 0.053691] |
| First-order gradient × delta | 0.896159 | [0.879578, 0.912739] | 0.357586 | 0.164388 [0.065403, 0.263373] |
| Full protection | 0.925781 | [0.925781, 0.925781] | 0.260861 | — |

The lowest-accuracy MSB result is trial 3 (schedule seed `930003`, schedule
SHA-256 beginning `2373e8`):

| Policy | Accuracy | Cross-entropy | Suppressed events |
|---|---:|---:|---:|
| No protection | 0.269531 | 6.243240 | 0 |
| M2 MSB-first | 0.296875 | 5.554520 | 16 |
| M2 random | 0.273438 | — | — |
| First-order gradient × delta | 0.902344 | 0.325501 | 13 |
| Full protection | 0.925781 | — | 132 |

The schedule contains 132 logical bit events: 129 in `dense0`, three in
`dense1`; by encoded position, counts are 23, 28, 31, 20, 13, and 17 for
positions 5 through 0 respectively. The MSB and gradient masks suppress 16 and
13 scheduled events. Two suppressed events overlap; 11 are gradient-only and
14 are MSB-only.

A bounded diagnostic replay injected each of the 132 trial-3 events in
isolation on the same 256 evaluation samples and restored state after every
event. All three output-layer events were gradient-only protected:

| Bit | Exact delta | Isolated accuracy damage | Isolated loss increase |
|---|---:|---:|---:|
| `dense1/kernel[110]#b5` | +4 | 0.312500 | 1.463690 |
| `dense1/kernel[125]#b4` | +2 | 0.265625 | 0.729667 |
| `dense1/kernel[121]#b5` | -4 | 0.132812 | 1.711510 |

The most damaging MSB-only protected event caused 0.011719 isolated accuracy
damage; most MSB-only events had zero isolated accuracy damage on this finite
sample. This explains a plausible mechanism for the catastrophic paired
difference, but isolated effects are not additive and this diagnostic does not
prove a general optimal policy.

The complete scheduled symmetric difference is recorded here using
`parameter[flat_index]#bit` identities:

- MSB-only: `dense0/kernel[302]#b5`, `[1513]#b5`, `[1686]#b5`,
  `[2571]#b5`, `[3631]#b5`, `[4512]#b5`, `[6347]#b5`, `[7048]#b5`,
  `[8131]#b5`, `[10828]#b5`, `[11946]#b5`, `[14685]#b5`, `[14888]#b5`,
  and `[15227]#b5`.
- Gradient-only: `dense0/kernel[6692]#b4`, `[9239]#b4`, `[10258]#b4`,
  `[10476]#b4`, `[13970]#b4`, `[16386]#b3`, `[17631]#b5`,
  `[18439]#b5`; and `dense1/kernel[110]#b5`, `[121]#b5`, `[125]#b4`.

## Mask composition

At 10% coverage every mask contains exactly 15,270 of 152,700 eligible bits.

| Mask | `dense0` | `dense1` | Kernel | Bias | Position counts b5,b4,b3,b2,b1,b0 |
|---|---:|---:|---:|---:|---|
| M2 MSB | 15,270 | 0 | 15,238 | 32 | 15,270,0,0,0,0,0 |
| M2 gradient × delta | 14,220 | 1,050 | 15,175 | 95 | 7,798,4,848,2,038,487,74,25 |
| M2 random | 15,045 | 225 | — | — | approximately uniform |

The M2 MSB mask therefore reflects canonical parameter ordering within its
partially selected sign-bit tier. It is not a generic result for MSB-primary
ranking. M3 gives the secondary ordering an explicit policy identity.

## Sample provenance and quantized representation

The M2 adapter implementation used `x_train[50000:...]` for ranking and
`x_test[:...]` for final evaluation, so the executed arrays do not overlap.
However, the M2 configuration and manifest label both through the single
ambiguous string `v2_test_first_n` and do not record explicit indices and
sample hashes. The comparison is therefore not known to have selection
leakage from code reconstruction, but the finalized bundle does not prove
independence by itself. M3 records role-specific split names, exact indices,
hashes, and enforces disjointness for confirmatory configurations.

The modern adapter loads checkpoint parameters that are exactly representable
under `q<6,2,1>`, checks encode/decode round trips, and fault evaluation XORs
the encoded word before decoding the changed value used by inference. M2 clean
inference therefore uses the same quantized values as injection. This is a
logical encoded-bit model, not a physical upset-rate calibration.
