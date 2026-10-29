from dataclasses import replace
from pathlib import Path

from netsurf.empirical import stratified_candidates
from netsurf.identities import BitIdentity
from netsurf.parallel import ExecutionResources
from netsurf.study_config import StudyEstimator, StudyReplications, load_study
from netsurf.study_engine import execute_study


ROOT = Path(__file__).resolve().parents[1]


def test_m91_stratified_design_has_four_candidates_per_kernel_bit_cell():
    identities = tuple(
        BitIdentity(f"tflite_tensor_{tensor:04d}/kernel", flat_index, bit_index)
        for tensor in range(10)
        for flat_index in range(5)
        for bit_index in range(8)
    )
    selected = stratified_candidates(
        identities, 320, seed=910101, scores=None, quantile_count=1
    )
    counts = {}
    for candidate in selected:
        key = (candidate.identity.parameter, candidate.identity.bit_index)
        counts[key] = counts.get(key, 0) + 1
        assert candidate.inclusion_rule == "stratified_layer_class_bit_score_quantile"
        assert candidate.sampling_probability == 0.8
    assert len(counts) == 80
    assert set(counts.values()) == {4}
    assert len({candidate.identity for candidate in selected}) == 320


def test_m91_configs_validate_and_preserve_distinct_replication_factors():
    validation = load_study(
        ROOT / "config/experiments/m91_resnet8_estimator_validation_stratified320_v1.toml",
        check_inputs=False,
    )
    assert validation.schema_version == "3.3.0"
    assert validation.candidates.strategy == "stratified"
    assert validation.candidates.count == 320
    assert validation.candidates.score_estimator is None
    assert validation.candidates.score_quantiles == 1
    assert validation.samples.ranking_count == 64
    assert validation.samples.empirical_count == 64
    assert validation.samples.evaluation_count == 200

    protection = load_study(
        ROOT / "config/experiments/m91_resnet8_static_protection_pilot_v1.toml",
        check_inputs=False,
    )
    assert protection.protection.coverages == (0.125, 0.25)
    assert protection.protection.probabilities == (0.0001,)
    assert protection.execution.trials == 8
    assert protection.replications.count == 4
    assert protection.replications.estimator_seed_stride == 1009
    assert protection.replications.fault_seed_stride == 100003
    assert protection.replications.hutchinson_seed_stride == 0


def test_schema33_advances_random_mask_seed_independently_of_hutchinson_seed():
    base = load_study(
        ROOT / "config/experiments/numpy_single_bit.toml", check_inputs=False
    )
    random = StudyEstimator(
        "random", "seeded-uniform-score/v1", 71, 16, "random_reference"
    )
    configuration = replace(
        base,
        schema_version="3.3.0",
        estimators=(random,),
        candidates=replace(base.candidates, strategy="uniform", count=4, seed=83),
        replications=StudyReplications(2, 0, 5000, 0, 0, 17),
    )
    outcome = execute_study(
        configuration, resources=ExecutionResources(workers=1, threads_per_worker=1)
    )
    seeds = [member.estimator_outputs[0].metadata["seed"] for member in outcome.replications]
    assert seeds == [71, 88]
    assert outcome.scientific_payload["replication_plan"]["estimator_seed_stride"] == 17
    assert outcome.scientific_payload["replication_plan"]["hutchinson_seed_stride"] == 5000
    for member in outcome.replications:
        overall = next(row for row in member.ranking_metrics if row["scope"] == "overall")
        assert overall["bootstrap_replicates"] == 1000
        assert overall["bootstrap_design"].endswith("sample-limited")
        assert overall["spearman_positive_damage_ci_low"] <= overall["spearman_positive_damage_ci_high"]


def test_schema32_resolved_document_is_not_mutated_by_schema33_extension():
    historical = load_study(
        ROOT / "config/experiments/m9_resnet8_static_estimator_validation_v1.toml",
        check_inputs=False,
    )
    assert historical.schema_version == "3.2.0"
    assert "estimator_seed_stride" not in historical.resolved_document()["replications"]
