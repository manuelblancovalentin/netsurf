"""Model-free exact W8A8 universe, field, and whole-population mask checks."""

from types import SimpleNamespace
import numpy as np
import pytest

from netsurf.w8a8_assets import W8A8AssetBlocked, load_ready_w8a8
from netsurf.w8a8_faults import W8A8Universe, bernoulli_field, FixedMask, MSBMask
from netsurf.w8a8_rank import positive_float_keys, random_hash_keys, random_chunks, write_top_masks, gradient_score_chunks, numerical_jump_chunks
from netsurf.manifests import sha256_file


def _universe():
    q = {"tensors": [{"module": "a", "shape": [2, 3], "qweight_sha256": "a", "scale_sha256": "b"},
                     {"module": "lm_head", "shape": [1, 2], "qweight_sha256": "c", "scale_sha256": "d"}]}
    return W8A8Universe(SimpleNamespace(quantization=q, eligible_bits=64))


def test_exact_module_order_and_encoded_bit_identity():
    u = _universe()
    assert u.bits == 64 and u.coordinate(47).module == "a"
    assert u.coordinate(48).module == "lm_head"
    assert u.coordinate(u.ordinal("lm_head", 1, 7)).bit == 7
    with pytest.raises(ValueError, match="outside"):
        u.coordinate(64)
    with pytest.raises(W8A8AssetBlocked):
        W8A8Universe(SimpleNamespace(quantization={"tensors": u.entries}, eligible_bits=63))


def test_geometric_gaps_are_reproducible_and_match_ber():
    a = bernoulli_field(64, .1, 17)
    assert a == bernoulli_field(64, .1, 17)
    assert a.ordinals == tuple(sorted(set(a.ordinals)))
    assert not bernoulli_field(64, 0, 17).ordinals
    counts = [len(bernoulli_field(1000, .02, seed).ordinals) for seed in range(300)]
    assert abs(np.mean(counts) - 20) < 1
    with pytest.raises(ValueError, match="bound"):
        bernoulli_field(1000, .5, 4, max_events=1)


def test_mask_budget_and_exact_streaming_topk(tmp_path):
    n = 64
    assert sum(MSBMask(n, 8).contains(i) for i in range(n)) == 8
    assert all(i % 8 == 7 for i in range(n) if MSBMask(n, 8).contains(i))
    assert FixedMask(n, frozenset({1, 2})).budget == 2
    raw = np.array([2., 2., 3., 0., 1., 3., 3., 1.] * 8, dtype=np.float32)
    def chunks():
        for start in range(0, n, 16):
            yield start, positive_float_keys(raw[start:start+16])
    result = write_top_masks(chunks, n, (8, 16), tmp_path / "masks", "jump")
    expected = sorted(range(n), key=lambda i: (-raw[i], i))
    for k in (8, 16):
        assert [i for i in range(n) if result[k].contains(i)] == sorted(expected[:k])
        assert result[k].path.stat().st_size == 8
    with pytest.raises(ValueError, match="existing"):
        write_top_masks(chunks, n, (8,), tmp_path / "masks", "jump")
    assert np.array_equal(random_hash_keys(0, 64, 9), random_hash_keys(0, 64, 9))
    assert not np.array_equal(random_hash_keys(0, 64, 9), random_hash_keys(0, 64, 10))
    random_mask = write_top_masks(lambda: random_chunks(n, 12, bits_per_chunk=16), n,
                                  (8,16), tmp_path / "random", "random", key_dtype=np.uint64)
    priorities = np.concatenate([keys for _, keys in random_chunks(n, 12, bits_per_chunk=16)])
    order = sorted(range(n), key=lambda i: (-int(priorities[i]), i))
    assert [i for i in range(n) if random_mask[8].contains(i)] == sorted(order[:8])


def test_signed_surrogate_bit_delta_and_positive_part(tmp_path):
    import json
    class Tensor:
        def __init__(self, array): self.array = array
        def reshape(self, *_): return self
        def detach(self): return self
        def cpu(self): return self
        def numpy(self): return self.array
    u = W8A8Universe(SimpleNamespace(quantization={"tensors": [{"module": "m", "shape": [1, 1],
        "qweight_sha256": "a", "scale_sha256": "b"}]}, eligible_bits=8))
    modules = {"m": SimpleNamespace(qweight=Tensor(np.array([0], dtype=np.int8)),
                                    scale=Tensor(np.array([.5], dtype=np.float32)))}
    g = tmp_path / "g.f32"; a = tmp_path / "a.f32"
    np.array([0], dtype="<f4").tofile(g)
    np.array([2], dtype="<f4").tofile(a)
    (tmp_path / "manifest.json").write_text(json.dumps({"version": "w8a8-decoded-code-surrogate-moments/v1",
        "universe_sha256": u.digest, "tensors": {"m": {"codes": 1, "mean_g": g.name,
        "mean_abs_g": a.name, "mean_g_sha256": sha256_file(g), "mean_abs_g_sha256": sha256_file(a)}}}))
    absolute = next(gradient_score_chunks(modules, u, tmp_path, "per_example_absolute"))[1].view(np.float32)
    positive = next(gradient_score_chunks(modules, u, tmp_path, "per_example_positive"))[1].view(np.float32)
    global_score = next(gradient_score_chunks(modules, u, tmp_path, "global_first_order"))[1].view(np.float32)
    assert absolute[0] == 1 and absolute[7] == 128
    assert positive[0] == .5 and positive[7] == 64
    assert np.array_equal(global_score, np.zeros(8, dtype=np.float32))
    jump = next(numerical_jump_chunks(modules, u))[1].view(np.float32)
    assert jump[0] == .5 and jump[7] == 64


def test_ready_record_and_quantization_identity_rejection(tmp_path):
    import json
    source = json.load(open("/projects/kona-assets/manuelbv/NETSURF_ASSET_READINESS.json"))
    target = next(x for x in source["targets"] if x["target_id"] == "distilbert_sst2_int8")
    path = tmp_path / "ready.json"
    target["status"] = "PREPARED_AWAITING_APPROVAL"
    path.write_text(json.dumps(source))
    with pytest.raises(W8A8AssetBlocked, match="READY"):
        load_ready_w8a8(path, "distilbert_sst2_int8", verify_files=False, require_interpreter=False)
    target["status"] = "READY"
    next(x for x in target["model"]["files"] if x["path"] == "quantization_manifest.json")["sha256"] = "0" * 64
    path.write_text(json.dumps(source))
    with pytest.raises(W8A8AssetBlocked, match="quantization manifest hash"):
        load_ready_w8a8(path, "distilbert_sst2_int8", verify_files=False, require_interpreter=False)


def test_checksum_bound_scientific_member(tmp_path):
    import json
    from netsurf.w8a8_campaign import _write_bound, _read_bound
    path = tmp_path / "member.json"
    _write_bound(path, {"plan_sha256": "a" * 64, "mean_nll": 1.})
    assert _read_bound(path)["mean_nll"] == 1.
    item = json.loads(path.read_text())
    item["mean_nll"] = 2.
    path.write_text(json.dumps(item))
    with pytest.raises(ValueError, match="checksum"):
        _read_bound(path)
