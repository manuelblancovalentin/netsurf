# KWS higher-BER conditional-selection pilot

This is a separately frozen **exploratory engineering** study. It does not
replace, extend the confirmatory family of, or rewrite the finalized M9.7
0.1/0.3/1% campaign. The same 12 M9.6 status-stratified K=6 candidate pools,
the same verified 22,016-of-176,128-bit global S5 mask, and the same 64
selection and 512 evaluation clips are used. All those clips have previously
been exposed to fault outcomes. Selection and final fault-field seeds are new,
disjoint from each other and from the earlier three-BER seed namespace.

The four logical Bernoulli bit-error probabilities are 0.02, 0.05, 0.075,
and 0.1. They correspond to 3,523, 8,806, 13,210, and 17,613 **expected
proposed** flips per 176,128-bit field, and 3,082, 7,706, 11,558, and 15,411
expected applied outside-mask flips. These are software stress conditions,
not calibrated radiation rates. Each arm uses two independent selection
seed streams of four outside fields per pool (eight fields), all 64 exact
candidate states, and 32 independent paired final master fields per pool.
Every mask protects exactly two candidate bits, preserving the 22,016-bit
whole-network budget. The exact isolated selector and original S5 mask are
retained as comparators. Final CE differences are paired within each field.

The purpose is to determine whether there is a non-saturated high-BER region
where the bounded conditional selector changes decisions and produces a
practically meaningful held-out effect. No endpoint was selected after
examining these new outcomes. The nominal pool intervals are descriptive,
not a prospective confirmation test; twelve pools and 32 fields per pool
give limited precision. A later confirmation study needs a newly frozen
sample/pool/field plan. A local K=6 effect is not a whole-network optimum.

## Frozen work plan and resource allowance

| Quantity | Each BER | All four BERs |
|---|---:|---:|
| Pools | 12 | 48 pool-BER members |
| Selection outside fields | 96 | 384 |
| Selection state requests | 6,228 | 24,912 |
| Selection clip inferences | 398,592 | 1,594,368 |
| Final paired master fields | 384 | 1,536 |
| Final state requests, conservative upper bound | 4,608 | 18,432 |
| Final clip inferences, conservative upper bound | 2,359,296 | 9,437,184 |

This is at most 43,344 state requests and 11,031,552 clip inferences, plus
bounded clean-reference loads. The completed older three-BER study used
33,046,272 planned clip inferences and 1.3 GiB of scientific evidence; its
measured engine time was about 11.5 hours. At comparable throughput this
pilot would take about 3.8 hours. High-BER event-list processing and current
host contention are not yet measured, so allow **5–10 hours** and up to
**2 GiB** of evidence; this is an estimate, not an execution deadline or a
formal power calculation. The 0.1 arm should be benchmarked on one member
before claiming the under-12-hour target is secure.

## Exact execution and status

The source must be committed before freezing, because the frozen pilot
manifest binds the Git revision. The pinned TensorFlow Lite 2.14.1
`BUILTIN_REF` interpreter is used for all model requests; the separate
NetSuRF analysis interpreter is used for offline aggregation. No original
experiment or asset lock is rewritten.

```bash
cd /projects/netsurf/manuelbv/workspace/implementation-worktrees/kws-ber-transition-pilot
ML=/projects/kona-assets/manuelbv/.venv-mlperf/bin/python
M=workspace/runs/kws_higher_ber_pilot_v1/campaign.json

$ML -m netsurf multiber freeze --higher-ber-pilot \
  --output workspace/runs/kws_higher_ber_pilot_v1
$ML -m netsurf multiber validate "$M"
# One member can be timed without refreezing or duplicating it later:
$ML -m netsurf multiber run "$M" --phase selection --ber 0.1 \
  --pools pool_000 --no-progress
# Complete, recoverable at finalized pool boundaries:
bash scripts/run_kws_higher_ber_pilot_once.sh
```

The one-shot runner verifies the frozen plan, finishes selection in every
arm, seals every mask before any final evaluation, evaluates all paired
final fields, and finalizes an offline analysis. It fails closed on an
interrupted member attempt; inspect the attempt and bundle before any
recovery action. The stage and raw log are under the pilot run root.

```bash
/mnt/raid0/ml/conda_envs/manuelbv/envs/netsurf/bin/python \
  scripts/watch_kws_higher_ber_pilot.py
# Or a single plain-text/JSON status check:
/mnt/raid0/ml/conda_envs/manuelbv/envs/netsurf/bin/python \
  scripts/watch_kws_higher_ber_pilot.py --once
```

The watcher is read-only. Progress counts **verified saved pool-phase
receipts**. It cannot know the exact percentage inside an active pool;
the live Rich CLI in the tmux runner supplies that finer view. ETA begins
only after measured pool runtimes exist and is labeled rough. A high-BER
near-chance task plateau must be reported as saturation, not as evidence
that conditional selection improves monotonically.
