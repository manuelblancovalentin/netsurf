# M9.5.5 efficient held-out protection-regret estimation

## Separation of selection and comparison

Schema 6.3 performs two different operations:

1. **Restricted selection.** Every `2^K` candidate state is evaluated for each
   independently seeded model-selection outside field. Exact Bernoulli
   integration ranks all fixed-budget masks. Candidate selection never reads
   final outcomes.
2. **Frozen-mask comparison.** The selected, exact-isolated, and original S5
   masks are frozen. Additional final master fields estimate only their paired
   expected-risk contrasts through exact TFLite evaluations.

The old full-enumeration final mode remains enabled with a small field count as
a numerical reference and post-hoc diagnostic. Its post-hoc best mask is not a
confirmation result.

## Exact paired-swap identity

For a protected bit `a`, unprotected bit `b`, current mask `P`, and
`P' = (P - {a}) union {b}`:

`J_p(P') - J_p(P) = p E_G[R(G union {a}) - R(G union {b})]`.

`G` is independently Bernoulli on all other unprotected eligible coordinates.
For masks requiring several exchanges, sorted removals are paired with sorted
additions. Every step is a legal cardinality-preserving swap. Step contributions
are summed within each master field before confidence intervals are calculated,
so covariance along the path is retained.

All risks use authoritative TensorFlow 2.14.1 TFLite `BUILTIN_REF` inference
and exact INT8 XOR states. No Hessian, Taylor truncation, continuous-weight
approximation, or alternate arithmetic is used.

The following frozen comparisons are supported:

- `restricted_selected_vs_original_s5`;
- `restricted_selected_vs_exact_isolated`;
- `exact_isolated_vs_original_s5`.

The sign is always target minus baseline. Negative values favor the target.

## Mathematical and response-table validation

Small-space tests integrate the complete Bernoulli fault space for additive,
pairwise, beneficial, third-order, and fourth-order pseudo-Boolean risks. They
verify orientation, zero BER, unequal bit roles, multi-swap telescoping, fixed
cardinality, and fixed-context suppression.

For every retained full-enumeration KWS field, NetSuRF also reconstructs each
frozen-mask contrast by telescoping directly through the saved response table
and compares it with direct exact mask-risk integration. This is an exact
same-table equality. The separate paired Monte Carlo estimate uses independent
master fields and can differ through finite sampling.

## Statistical design

The K=8 configuration has 28 eligible budget-two masks. It increases model-
selection evidence from two outside fields in M9.5.4 to eight fields arranged
as four explicit replicate seeds with two fields per seed. It records:

- the winner for each individual field;
- the cumulative winner after each field;
- the cumulative best-versus-second risk margin;
- cumulative winner switches and per-mask win counts.

These diagnostics expose selection sensitivity but do not constitute a formal
power calculation. Eight fields are an engineering starting point, not a claim
of stable selection. If a pilot shows persistent cumulative switches or a
margin small relative to field variability, a larger prespecified model-
selection run should be created without inspecting final outcomes.

The paired final study uses 64 independently seeded master fields. Each field
is one statistical observation after the complete path is summed. Model-
selection fields, input examples, swap steps, and individual TFLite calls are
not additional final replications. Intervals remain conditional on the frozen
checkpoint and final input set.

M9.5.3 exposed final relative indices 0--5 and M9.5.4 exposed 6--11. Schema
6.3 begins final evaluation at relative index 12 (original KWS test index 3060).

## Operation counts

The bounded K=6 validation requests at most:

- 38 conditional-damage evaluations;
- 199 restricted-oracle evaluations: 7 isolated states plus three exact
  response tables of 64 states;
- 48 paired-swap TFLite requests across four master fields and three
  comparisons;
- 285 total authoritative requests before cross-comparison cache reuse;
- 45 analytical mask-risk integrations and 315 analytical pair differences.

The K=8 researcher configuration requests at most:

- 210 conditional-damage evaluations;
- 2,313 restricted-oracle evaluations: 9 isolated states plus nine tables of
  256 states (eight selection and one final reference);
- 768 paired-swap requests for 64 master fields and three comparisons;
- 3,291 total authoritative requests before cache reuse;
- 252 analytical mask-risk integrations and 3,402 pair differences.

Actual unique TFLite inference count is recorded because comparisons can share
identical schedules. Analytical response-table reuse adds no model inference.
Based on the 93-second M9.5.4 K=6 smoke and the larger K=8 sample partitions,
the K=8 configuration should be budgeted approximately 45--90 CPU minutes.
This is a planning range, not a guaranteed runtime.

## Commands

From the M9.5.5 worktree, validate the bounded configuration:

```bash
/projects/kona-assets/manuelbv/.venv-mlperf/bin/python -m netsurf conditional validate \
  config/examples/m955_kws_paired_swap_k6_validation.toml --resolved
```

Validate the researcher K=8 configuration:

```bash
/projects/kona-assets/manuelbv/.venv-mlperf/bin/python -m netsurf conditional validate \
  config/examples/m955_kws_s5_context_k8.toml --resolved
```

After inspecting resolved counts, run K=8 interactively in tmux:

```bash
NETSURF_WORKSPACE="$PWD/workspace" \
  /projects/kona-assets/manuelbv/.venv-mlperf/bin/python -m netsurf conditional run \
  config/examples/m955_kws_s5_context_k8.toml
```

Do not tune field counts or masks after inspecting final outcomes. A changed
design requires a new named configuration and independent final seeds.
