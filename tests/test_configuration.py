from pathlib import Path
import tempfile
import tomllib

from netsurf.experiment_config import (
    EXPERIMENT_SCHEMA_VERSION,
    ConfigurationError,
    UnsupportedSchemaVersion,
    load_experiment,
    load_schema,
    template_text,
    validate_experiment_document,
)


ROOT = Path(__file__).resolve().parents[1]


def test_modern_schema_version_survives_resolution():
    configuration = load_experiment(
        Path(__file__).resolve().parents[1] / "config/experiments/numpy_reference.toml"
    )
    assert configuration.schema_version == "2.0.0"
    assert configuration.resolved_document()["schema_version"] == "2.0.0"
    assert configuration.resolved_document()["source_configuration"].endswith(
        "config/experiments/numpy_reference.toml"
    )
    assert configuration.scientific_protocol == "quantized-weight-fault/v2"


def test_modern_single_and_multiple_estimator_examples_validate():
    single = load_experiment(ROOT / "config/experiments/mnist_msb.toml", check_inputs=False)
    comparison = load_experiment(
        ROOT / "config/experiments/mnist_ranker_comparison.toml", check_inputs=False
    )
    assert [value.name for value in single.estimators] == ["msb_first"]
    assert [value.name for value in comparison.estimators] == [
        "msb_first",
        "random",
        "first_order_gradient_delta",
    ]
    assert single.benchmark.injectable_parameters == ("kernels", "biases")
    assert comparison.fault.persistence == "persistent_trial"


def test_packaged_templates_validate_without_loading_frameworks():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for name in ("template", "smoke"):
            path = root / f"{name}.toml"
            path.write_text(template_text(name), encoding="utf-8")
            configuration = load_experiment(path, check_inputs=False)
            assert configuration.schema_version == EXPERIMENT_SCHEMA_VERSION
            assert configuration.output.root.name == "runs"
            assert configuration.benchmark.checkpoint.is_absolute()


def test_unsupported_schema_version_fails_clearly():
    document = tomllib.loads(template_text("smoke"))
    document["schema_version"] = "9.0.0"
    try:
        validate_experiment_document(document, Path("/tmp/config.toml"), check_inputs=False)
    except UnsupportedSchemaVersion as error:
        assert "9.0.0" in str(error)
        assert EXPERIMENT_SCHEMA_VERSION in str(error)
    else:
        raise AssertionError("unsupported schema version was accepted")


def test_unsupported_estimator_is_not_silently_ignored():
    document = tomllib.loads(template_text("smoke"))
    document["estimators"][0]["name"] = "future_magic_estimator"
    try:
        validate_experiment_document(document, Path("/tmp/config.toml"), check_inputs=False)
    except ConfigurationError as error:
        assert "future_magic_estimator" in str(error)
    else:
        raise AssertionError("unsupported estimator was accepted")


def test_exact_mnist_adapter_requires_kernel_bias_population():
    document = tomllib.loads(
        (ROOT / "config/experiments/numpy_reference.toml").read_text(encoding="utf-8")
    )
    document["benchmark"]["adapter"] = "mnist_lite_exact_kb_v1"
    document["benchmark"]["injectable_parameters"] = ["kernels", "biases"]
    configuration = validate_experiment_document(
        document, Path("/tmp/config.toml"), check_inputs=False
    )
    assert configuration.benchmark.adapter == "mnist_lite_exact_kb_v1"

    document["benchmark"]["injectable_parameters"] = ["all_trainable"]
    try:
        validate_experiment_document(document, Path("/tmp/config.toml"), check_inputs=False)
    except ConfigurationError as error:
        assert "alpha/beta" in str(error)
    else:
        raise AssertionError("unresolved auxiliary tensors were accepted as exact fault targets")

    document["benchmark"]["injectable_parameters"] = ["kernels", "biases"]
    document["benchmark"]["quantization"] = "q<8,3,1>"
    try:
        validate_experiment_document(document, Path("/tmp/config.toml"), check_inputs=False)
    except ConfigurationError as error:
        assert "q<6,2,1>" in str(error)
    else:
        raise AssertionError("an identity-changing codec was accepted")


def test_versioned_schema_documents_are_exercised():
    experiment = load_schema("experiment")
    manifest = load_schema("manifest")
    summary = load_schema("summary")
    measurements = load_schema("measurements")
    assert experiment["properties"]["schema_version"]["const"] == "1.0.0"
    assert manifest["properties"]["schema_version"]["const"] == "1.0.0"
    assert summary["properties"]["schema_version"]["const"] == "1.0.0"
    assert measurements["schema_version"] == "1.0.0"
    modern_experiment = load_schema("experiment", "v2")
    modern_measurements = load_schema("measurements", "v2")
    assert modern_experiment["properties"]["schema_version"]["const"] == "2.0.0"
    assert modern_measurements["schema_version"] == "2.0.0"
    assert "runtime_seconds" in modern_measurements["tables"]["estimator_runtime.csv"]["columns"]
    study_experiment = load_schema("experiment", "v3")
    study_measurements = load_schema("measurements", "v3")
    assert study_experiment["properties"]["schema_version"]["const"] == "3.0.0"
    assert study_measurements["schema_version"] == "3.0.0"
    assert "signed_loss_difference" in study_measurements["tables"]["empirical_bits.csv"]["columns"]
    assert "p_value_holm" in study_measurements["tables"]["paired_comparisons.csv"]["columns"]
