from pathlib import Path

from netsurf.adapters.numpy_reference import NumpyReferenceAdapter
from netsurf.experiment_config import load_experiment
from netsurf.ordering import compose_ordering, standard_policy_definition
from netsurf.protection import ProtectionPolicy
from netsurf.quantization import BitCodec, FixedPointFormat
from netsurf.ranking import FirstOrderEstimator, MSBEstimator, RandomEstimator


ROOT = Path(__file__).resolve().parents[1]


def _reference():
    configuration = load_experiment(ROOT / "config/experiments/numpy_reference.toml")
    codec = BitCodec(FixedPointFormat.parse(configuration.benchmark.quantization))
    adapter = NumpyReferenceAdapter(configuration.benchmark.checkpoint, configuration.benchmark.dataset, codec)
    adapter.load()
    return adapter, codec


def test_composable_msb_tiers_and_layer_directions():
    adapter, _ = _reference()
    identities = adapter.bit_identities()
    ranking = MSBEstimator().estimate(identities)
    specifications = adapter.parameter_specs()
    ordered = compose_ordering("ordered", ranking, identities, specifications, ("bit_significance_desc", "canonical"))
    reversed_ = compose_ordering("reversed", ranking, identities, specifications, ("bit_significance_desc", "reverse_canonical"))
    input_first = compose_ordering("input", ranking, identities, specifications, ("bit_significance_desc", "layer_input_first", "stable_identity"))
    output_first = compose_ordering("output", ranking, identities, specifications, ("bit_significance_desc", "layer_output_first", "stable_identity"))
    assert [value.bit_index for value in ordered.identities] == sorted([value.bit_index for value in identities], reverse=True)
    assert ordered.identities[0] != reversed_.identities[0]
    assert input_first.identities[0].parameter.startswith("dense0/")
    assert output_first.identities[0].parameter.startswith("dense1/")
    assert len(set(ordered.identities)) == len(identities)


def test_random_tie_breaking_is_seeded_and_stays_inside_primary_tier():
    adapter, _ = _reference()
    identities, specs = adapter.bit_identities(), adapter.parameter_specs()
    ranking = MSBEstimator().estimate(identities)
    first = compose_ordering("a", ranking, identities, specs, ("bit_significance_desc", "random"), seed=11)
    repeated = compose_ordering("b", ranking, identities, specs, ("bit_significance_desc", "random"), seed=11)
    different = compose_ordering("c", ranking, identities, specs, ("bit_significance_desc", "random"), seed=12)
    assert first.identities == repeated.identities
    assert first.identities != different.identities
    assert [value.bit_index for value in first.identities] == sorted([value.bit_index for value in identities], reverse=True)


def test_all_msb_variants_have_identical_complete_tier_masks():
    adapter, codec = _reference()
    identities, specs = adapter.bit_identities(), adapter.parameter_specs()
    inputs, labels = adapter.dataset(8, role="ranking")
    gradients = adapter.gradients(inputs, labels)
    deltas = {name: codec.perturbation_matrix(value) for name, value in adapter.values().items()}
    rankings = {
        "msb_first": MSBEstimator().estimate(identities),
        "first_order_gradient_delta": FirstOrderEstimator().estimate(identities, gradients, deltas),
        "random": RandomEstimator(5).estimate(identities),
    }
    masks = []
    for name in ("msb+ordered", "msb+reversed", "msb+random", "msb+layer_input_first", "msb+layer_output_first", "msb+gradient"):
        criteria, estimator = standard_policy_definition(name)
        ordering = compose_ordering(name, rankings[estimator], identities, specs, criteria, seed=7 if "random" in criteria else None)
        masks.append(ProtectionPolicy.ordered(ordering, 1.0 / codec.format.total_bits).protected)
    assert all(mask == masks[0] for mask in masks[1:])
    assert all(value.bit_index == 5 for value in masks[0])
    partial = [
        ProtectionPolicy.ordered(
            compose_ordering(name, rankings[standard_policy_definition(name)[1]], identities, specs, standard_policy_definition(name)[0], seed=7 if name == "msb+random" else None),
            0.10,
        ).protected
        for name in ("msb+ordered", "msb+reversed", "msb+random")
    ]
    assert len({frozenset(value) for value in partial}) == 3
