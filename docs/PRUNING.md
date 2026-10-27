# M5 structured pruning and reliability contract

## Method classification

NetSuRF separates three categories:

| Capability | Classification | Exact NetSuRF definition |
|---|---|---|
| Ten-stage, 14.9%-of-remaining width schedule | Reproduced from Vinck et al. | Integer targets approximate `initial_width * 0.851**stage`; the tenth target retains 20% when integer width permits. |
| Grouped first-order Taylor pruning | Architecture-specific Vinck/Torch-Pruning adaptation | For hidden neuron `j`, sum `abs(w * dL/dw)` over `dense0/kernel[:,j]`, `dense0/bias[j]`, and `dense1/kernel[j,:]`. Keras tensor layout and the explicit bias/outgoing dependency differ from the paper's PyTorch CNNs. |
| Magnitude control | NetSuRF control | Group L1 or L2 norm over the same complete hidden-neuron dependency group. |
| Independently trained width control | NetSuRF control | Fresh seeded Glorot initialization at the requested width, with the configured training mode; no source neurons are inherited. |
| `netsurf_reliability_aware` | New NetSuRF proposal | Min-max-normalized clean Taylor cost minus bit-susceptibility benefit minus released-resource benefit, with independently configurable nonnegative coefficients. Lower groups are removed first. |
| `netsurf_fault_augmented` fine-tuning | New NetSuRF proposal | Clean loss plus a separately weighted loss under exact quantized XOR samples, using a straight-through training gradient. It is not RAP. |

[Vinck et al.](https://arxiv.org/html/2610.03117v2) use ResNet-18 and
MobileNetV2 on EuroSAT, Torch-Pruning grouped Taylor importance, retraining
after every pruning step, and subsequent 8-bit QAT. Their fault experiment
selects one bit in an inference trace spanning 8-bit inputs/weights/outputs and
32-bit biases/accumulators; protected main memory is assumed. NetSuRF M5 copies
the pruning schedule and group-Taylor concept only. Its benchmark is an
explicit Dense--ReLU--Dense network and its normal fault contract remains
persistent Bernoulli XOR events in stored six-bit parameters.

The accessible primary page for [Ahsaei, Raji and Ghavami's
RAP](https://doi.org/10.1016/j.sysarc.2026.103829) states that the pruning
criterion combines weight magnitude with output changes caused by bit flips
and that fine-tuning is reliability-aware. It does not expose the equations,
normalization, fault sampling distribution, pruning granularity/selection
algorithm, or fine-tuning loss and schedule. Those details are required for a
faithful reproduction. Consequently, schema 5 deliberately rejects a method
named `rap`; the implemented method is explicitly
`netsurf_reliability_aware` and all three score components remain in checkpoint
provenance for ablation.

## Physical topology contract

Pruning removes an entire hidden dependency group. The resulting arrays are:

```text
dense0/kernel: [784, retained_hidden]
dense0/bias:   [retained_hidden]
dense1/kernel: [retained_hidden, 10]
dense1/bias:   [10]
```

No zero mask remains. Bit identities are re-enumerated in C order over the new
arrays. The checkpoint records parameter hashes, source checkpoint and dataset
hashes, every removed stage index, importance values/components, fine-tuning
seed/history, exact six-bit topology, parameter count, eligible-bit count,
dense MAC count, and encoded storage bytes. Checkpoints are created
exclusively, made read-only, and never overwrite their source.

MAC count is the two dense-kernel multiply-accumulate count. It is an operation
and exposure proxy only; no FIT or radiation claim follows without an explicit
physical dataflow, residence time, cross-section, and flux model.

## Schema 5.0 fields

The executable validator is `netsurf.pruning_config.load_pruning`; the JSON
schema is `netsurf/schemas/pruning/v5.schema.json`.

```toml
schema_version = "5.0.0"
scientific_protocol = "structured-hidden-neuron-pruning/v1"

[experiment]
name = "filesystem_safe_name"
description = "researcher-authored objective"

[source]
adapter = "mnist_quantized"
checkpoint = "/path/to/immutable/source.npz"
dataset = "/ml/datasets/.../mnist.npz"
quantization = "q<6,2,1>"

[calibration]
sample_count = 256
sample_offset = 0

[pruning]
method = "grouped_taylor" # group_magnitude | netsurf_reliability_aware | independent_width
schedule = "target_widths" # or paper_fraction
iterations = 1             # used by paper_fraction
step_fraction = 0.149       # used by paper_fraction
target_widths = [24]        # strictly descending; independent_width requires exactly one
magnitude_norm = "l1"       # l1 | l2

[reliability]
susceptibility_estimator = "first_order_gradient_delta"
susceptibility_reduction = "mean" # mean | sum | max
clean_weight = 1.0
susceptibility_weight = 1.0
cost_weight = 0.0

[fine_tuning]
enabled = false
mode = "quantization_aware" # standard | quantization_aware | netsurf_fault_augmented
epochs_per_stage = 0
batch_size = 128
learning_rate = 0.001
training_samples = 20000
clean_loss_weight = 1.0
fault_loss_weight = 0.0
fault_probability = 0.001
seed = 20261027

[output]
checkpoint = "/repository/workspace/benchmarks/my_pruned_model/checkpoint.npz"
```

`output.checkpoint` must resolve under the configured NetSuRF workspace. The
repository intentionally contains no maintained M5 research TOML.

## Evaluation interfaces

An M5 checkpoint uses the existing `mnist_quantized` adapter, so schema-2/3
static policies and schema-4 conditional/factorized searches consume the
physically reduced bit population without special policy code. Pruned-only
manifest fields add topology and pruning provenance; historical checkpoint
payloads remain unchanged.

The contracts are supplied by:

- clean/faulty accuracy and cross-entropy: existing experiment/study engines;
- clean-relative degradation and prediction-change probability for one fixed
  persistent state: `netsurf.reliability.evaluate_persistent_fault_state`;
- persistent per-bit BER and matched schedules: existing fault engine;
- conditional single-bit effects: schema-3 empirical and schema-4 conditional
  evaluators;
- fractional budgets: schema-2/3 policies;
- absolute budgets: schema-4 `initial_mask.budget_bits` or
  `budget_bits_list`;
- MAC, memory, parameter and eligible-bit counts: checkpoint topology and
  pruned-run manifest `benchmark.structured_pruning.topology`.

Per-inference Bernoulli and exact isolated-single-bit modes remain separate
optional experiments. They must not be described as the persistent stored-word
fault process used by ordinary protection trials.

## Commands

```bash
proj activate netsurf
netsurf pruning validate /path/to/research-authored-pruning.toml --resolved
netsurf pruning run /path/to/research-authored-pruning.toml
netsurf pruning inspect workspace/benchmarks/my_pruned_model/checkpoint.npz
```

After inspection, reference the new checkpoint from an existing schema-3 or
schema-4 experiment while keeping `adapter = "mnist_quantized"`. Start with a
bounded sample/trial count; pruning preparation may train only when explicitly
requested by the researcher's configuration.
