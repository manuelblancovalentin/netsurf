# M9.5.4 full-budget protection context and response-table reuse

## Verified source context

The maintained contextual examples identify, rather than infer, an existing
finalized KWS S5 mask:

- bundle: `workspace/runs/m93_kws_static_ber_sweep1000_v1/20261029T183410.488907Z-32c31070`;
- scientific payload SHA-256:
  `b7b725fd1c196f2d5896b67aba34dfe8aa864b8308875df098d48be6135975e0`;
- artifact: `artifacts/protection_masks.json`, SHA-256
  `2c5753a46bf09cb4f9f90bb73b80245ce70b2f57dfee52bf97c0e191e2e710b9`;
- key: `replication_0/per_example_absolute_global@0.125000`;
- population: 176,128 eligible KWS INT8 kernel bits;
- protected cardinality: 22,016 bits (12.5%).

The verifier also checks the finalized marker, manifest completion status,
checkpoint SHA-256, checksums inventory, identity uniqueness, and membership
in the current adapter's eligible universe. Asset locks are unchanged. The
618,880-bit population is ResNet-8, not KWS.

## Exact fixed-budget construction

Let `C` be the frozen candidate pool and `P0` the verified whole-network mask.
The engine fixes `P_out = P0 \\ C` and evaluates every candidate allocation

`P(Q) = P_out union Q`,

where `|Q| = |P0 intersect C|`. The pool must contain at least one protected
and one unprotected source bit. Thus the original static mask, exact isolated-
damage comparator, and model-selection restricted optimum all preserve the
same whole-network cardinality. The schema-6.2 control keeps `P_out` empty and
retains the earlier no-background-protection interpretation.

Outside-network Bernoulli fields exclude both `C` and `P_out`. Candidate-state
faults are enumerated separately. Existing exact encoded XOR, pairing,
restoration, and event-accounting contracts are unchanged.

## Response-table sufficiency

For each outside field `G`, the authoritative model evaluates every one of the
`2^K` candidate states once under fixed `P_out`. For a candidate protection
mask `Q`, states containing a protected candidate receive zero probability;
the remaining states receive their exact independent-Bernoulli mass. The same
saved table therefore yields, without further TFLite calls:

- every eligible fixed-budget mask risk;
- every candidate conditional marginal;
- every paired fixed-budget mask-risk difference.

Because the table is exhaustive over `C`, it retains additive, pairwise, and
all higher-order candidate interactions. Numerical tests compare these derived
risks with direct paired event-suppression evaluation, including beneficial and
pure higher-order responses. This equivalence is specific to the stated
independent Bernoulli law. A naturally unflipped bit is not generally
statistically interchangeable with a protected bit under correlated or
fixed-count laws.

## Partition audit

The pinned KWS TFRecord source order is a documented round-robin-block-16
order. The three fixed windows are contiguous regions of that order, not proof
of identical distributions. The adapter records full-window label counts,
label-bound hashes, source ordering, and the exact used-sample identities.

| Window | Original indices | Label-bound SHA-256 | Class-count range |
|---|---:|---|---:|
| Calibration | 1000--2023 | `41fa29a9e03d6cd63348ad1a6076b49162366a948dedf27d02e87bd9dd289d66` | 74--104 |
| Model selection | 2024--3047 | `d186393fcc6eb6d6ae3705c0ebbe3709a9034d345558e9a06787bb52260520f7` | 66--105 |
| Final evaluation | 3048--4889 | `9798cceda870b44cb6053198dfb80fc72d8e35195496a85c3dbf50929eb020c4` | 135--174 |

The M9.5.3 engineering smoke already exposed final-window relative indices
0--5 (original indices 3048--3053). Schema 6.2 therefore requires final
evaluation offset at least 6. Maintained examples begin at original index
3054. Calibration is descriptive, masks are selected on model-selection data,
and final data only compare frozen masks. The post-hoc final optimum remains a
selection-biased diagnostic.

## Bounded K=6 validation

The pinned TensorFlow 2.14.1 `BUILTIN_REF` environment completed the contextual
K=6 smoke with three final outside fields and 365 requested state evaluations.
It preserved the 22,016-bit global cardinality, verified restoration and
checksums, and emitted five response-table-derived records. The original
candidate mask was `(4, 5)`, the model-selection restricted mask was `(2, 4)`,
and the isolated comparator was `(0, 1)`.

The paired selected-minus-original final-risk mean was approximately
`-6.00e-05` with a 95% interval spanning zero. Three outside fields are an
engineering check only and provide no effectiveness conclusion. A clean-source
rerun and its final payload hash are recorded in the implementation handoff.

## Operation counts and cost

The exact count per BER is

`K + 1 + (B_select + B_final) * 2^K`

for the restricted stage, plus the unchanged conditional-damage stage. Both
maintained K=8 examples request 1,755 state evaluations (210 conditional and
1,545 restricted). The global context adds verification and set construction,
not TFLite inference. At budget two, each response table derives 28 mask risks
and 378 pairwise mask differences in memory. The earlier bounded K=6 contextual
run took about 93 seconds, including a one-time audit of all 4,890 source
labels. A conservative K=8 CPU estimate is 25--45 minutes for the maintained
32/64/128 sample counts; runtime scales linearly with samples and outside
fields and exponentially with `K`.

## Future guided-sampling boundary

Every response-table row now retains exact outside-fault identities, target
probability-law identity, BER, context identity, pairing key, candidate state,
and authoritative response. Supporting a future proposal distribution would
require a new reviewed sampler/law interface that additionally records target
and proposal probability masses (preferably log masses), proves full target
support, computes importance weights, and reports effective sample size and
weight diagnostics. Proposal and target seeds would remain separate.

M9.5.4 implements only independent Bernoulli sampling. Worst-case or guided
attack samples do not estimate Bernoulli expected risk without valid
probability correction.

## Researcher commands

Validate the retained no-context K=8 control:

```bash
<assets>/.venv-mlperf/bin/python -m netsurf conditional validate \
  config/examples/m954_kws_restricted_k8_control.toml --resolved
```

Validate the verified S5 12.5%-context K=8 exchange study:

```bash
<assets>/.venv-mlperf/bin/python -m netsurf conditional validate \
  config/examples/m954_kws_s5_context_k8.toml --resolved
```

After reviewing resolved counts, run either configuration from this worktree:

```bash
NETSURF_WORKSPACE="$PWD/workspace" \
  <assets>/.venv-mlperf/bin/python -m netsurf conditional run \
  config/examples/m954_kws_s5_context_k8.toml
```

The source S5 bundle must remain present and hash-valid. Copying or guessing a
mask path is not supported.
