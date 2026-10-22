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


def test_versioned_schema_documents_are_exercised():
    experiment = load_schema("experiment")
    manifest = load_schema("manifest")
    summary = load_schema("summary")
    measurements = load_schema("measurements")
    assert experiment["properties"]["schema_version"]["const"] == "1.0.0"
    assert manifest["properties"]["schema_version"]["const"] == "1.0.0"
    assert summary["properties"]["schema_version"]["const"] == "1.0.0"
    assert measurements["schema_version"] == "1.0.0"
