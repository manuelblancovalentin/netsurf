from netsurf.manifests import canonical_json_bytes, digest_object


def test_canonical_json_and_digest_ignore_mapping_order():
    first = {"b": [2, 3], "a": 1}
    second = {"a": 1, "b": [2, 3]}
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert digest_object(first) == digest_object(second)


def test_canonical_json_rejects_nonfinite_values():
    try:
        canonical_json_bytes({"invalid": float("nan")})
    except ValueError:
        pass
    else:
        raise AssertionError("non-finite JSON value was serialized")
