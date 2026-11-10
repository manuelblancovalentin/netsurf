# KWS independent-pool 5% BER validation

This separately frozen campaign tests whether exact bounded-candidate
conditional mask selection improves on exact isolated-bit selection at logical
Bernoulli BER 0.05. The 5% operating point was chosen after the exploratory
2–10% pilot; it is not an independently discovered BER transition.

The candidate pools are 32 fresh status-stratified uniform draws of two bits
protected by the verified global absolute-gradient mask and four unprotected
bits. Each pool retains the same two-bit candidate protection cardinality and
the original 22,016/176,128 global protected-bit cardinality. The frozen
comparison family has selected versus exact isolated as its primary paired
contrast; selected versus original static and isolated versus original static
are two Holm-adjusted secondary contrasts. A pool, not a field or clip, is the
principal replicate. The experiment cannot estimate a whole-network optimum.

Model selection uses the previously exposed original test indices 2152–2215.
The separate final cohort is original indices 4474–4857, 384 official clips
covering all twelve task labels; metadata-only exposure audit found no prior
fault-outcome exposure in known finalized bundles. This is conditional on the
documented exposure inventory; the researcher must disclose any manual or
out-of-root prior use. No final outcomes are inspected before masks are sealed.

Each pool uses two independent selection seed streams, four outside fields
each, and 32 independent paired final master fields. The exact K=6 response
table has 64 candidate states per outside field. Planned requests per pool:
519 selection states and at most 384 final states. For 32 pools this is
16,608 selection-state requests (1,062,912 clip inferences) and at most
12,288 final-state requests (4,718,592 clip inferences), plus model-load
reference gates. Total is at most 5,781,504 clip inferences. Scaling the
completed 11.0-million-clip high-BER pilot gives a provisional 3–6 hours;
allow 8 hours and 2 GiB for I/O and contention. Logical BER is not a physical
radiation rate; mitigation is ideal event suppression.

Run only after the source commit and frozen manifest validate:

```bash
cd /projects/netsurf/manuelbv/workspace/implementation-worktrees/kws-transition-validation
tmux -L netsurf-kws-transition new-session -d -s confirm \
  'bash scripts/run_kws_transition_validation_once.sh 2>&1 | tee workspace/runs/kws_transition_validation.log'
```

The runner is recoverable at verified completed pool receipts. An interrupted
unfinished pool attempt must be audited before re-execution; it is never
silently discarded. Final scientific conclusions require the finalized,
checksum-verified analysis, not live progress.
