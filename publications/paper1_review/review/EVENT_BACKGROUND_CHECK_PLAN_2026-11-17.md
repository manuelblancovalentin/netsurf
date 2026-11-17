# Check of the events-versus-background sketch (Fig. 2 d-f)

**Status:** frozen before the full run. Requested by the researcher on 2026-11-17 ("Yes, run it").

## Question

Panels (d-f) of the concept figure say, as a sketch: the low-bit background and the top-bit events grow together
with the fault rate; what changes is how many events reach one neuron: at most one at sparse rates (it stands out),
many at dense rates (their sum is itself close to a bell curve). Is that what happens in a real layer?

## Design (all fixed here)

- Model: the reference ResNet-8 (replication 1 of the main comparison), all nine CONV_2D layers. No other model.
- Inputs: the first 8 evaluation images of that replication. The clean inputs of each layer are held fixed, and
  only that layer's weights are faulted, so the change of each pre-activation is exactly linear in the weight
  displacements and splits exactly into a part from bits 0-5 (background) and a part from bits 6-7 (events).
  Effects of faults in earlier layers are not included.
- Fault model: each unprotected bit of the layer flips independently with probability p; protected bits never flip
  (ideal suppression, as in the fault-rate study and the model check).
- Masks: no protection, and the absolute-gradient mask at 12.5% from 1,024 calibration inputs (the saved mask of the
  main comparison).
- Rates: the eight rates of the main comparison, 1e-5 to 1e-1.
- Fields: 500 per layer, mask and rate; seed 20261117 with (layer, mask, rate, chunk).
- Before any fault is drawn, the script recomputes each layer's clean output and stops unless it agrees with the
  executable within one output step on at least 99.9% of the unsaturated outputs.
- Code: `workflows/event_background_check.py`, `netsurf/event_background.py`, `tests/test_event_background.py`
  (branch paper1-runR-v2-validated). Analysis: `workflows/event_background_analyze.py` merges the per-layer outputs
  and evaluates the statements below; it has no free choice.

## Statements and what counts as support

Evaluated on every layer and mask in which top bits are left unprotected; the figure uses the layer with the largest
fan-in (64 x 3 x 3 x 64, 576 weights per neuron), fixed here.

1. *They grow together.* The share of the change's second moment due to events differs by less than 0.02 between
   the eight rates. (The exact share is also computed from the moments.)
2. *Number of events.* Reported per rate: mean number of top-bit flips in a neuron's fan-in, and the probability of
   none, one, and two or more. No threshold; these are the numbers the caption will quote.
3. *One event stands out at sparse rates.* Where at least one event occurred, the median size of the event part is
   more than 3 standard deviations of the background at every rate at which the mean number of events is below 0.1.
4. *Many events add up to a bell curve.* The largest gap between the distribution of the standardized event part and
   a normal distribution falls as the rate grows, and is below 0.05 at p = 1e-1 on the figure layer without
   protection. The same gap is reported for the background and for the sum.

## What goes into the paper whatever the outcome

One appendix paragraph with a table (figure layer, both masks, eight rates) and one figure of measured histograms at
three rates. Fixed here: the rates shown are the lowest rate, the highest rate, and the rate at which the mean number
of events per neuron is closest to one. If a statement fails, the caption of Fig. 2 is changed to what was measured.
The result supports a sketch on one layer of one checkpoint; it is not a test of the statistical model.

## Disclosure

A code smoke was run before this plan: layer 0 (fan-in 27), p = 1e-3, 50 fields, both masks. Its output was read
(run time, reconstruction, and the printed statistics). The full run repeats that condition with the same first
chunk of fields.

## Resources

Nine processes (one per layer), four threads each, under 4 GB each, about 20 to 40 minutes on the shared server.
Output (write-once): `workspace/runs/paper1_event_background_v1/layer<k>/`; log
`workspace/runs/paper1_event_background_logs/run.log`.

## Amendment 1 (2026-11-17, during the run, before any result was read)

Three of the nine jobs stopped at the reconstruction check before drawing any fault: the second convolution of the
first block and the two 1 x 1 shortcut convolutions (kernels 0010, 0013, 0016). The executable computes the residual
addition that follows each of them in place, so the tensor that should hold the convolution's output holds the
result of the addition instead, and the check compared against the wrong values. The inputs of these three layers
are intact (each is also the input or the checked output of a layer that passed).

Change, for these three layers only: the reconstruction is checked on the output of the following addition. The
recomputed convolution output, clipped to the range its tensor can represent, is added to the other branch's stored
output and clipped to the range of the addition's output; it must agree with the executable within one step of the
convolution output plus one step of the addition output on at least 99.9% of the outputs that are not saturated.
Nothing else changes: same fields, seeds, masks, rates, statistics and statements. The six layers that passed keep
their outputs from the first launch (script hash recorded in their DONE files); the three are run with the amended
script. No statistic of any layer had been read when this was written; only job exit codes and the three error
messages.
