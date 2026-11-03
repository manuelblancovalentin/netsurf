import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import pytest

from netsurf.bf16_fp32_storage import widen_bf16_bytes
from netsurf.faults import FaultEvent, FaultModel, FaultSchedule
from netsurf.identities import BitIdentity
from netsurf.protection import ProtectionMask


def test_all_bf16_words_and_all_single_bit_transitions_widen_exactly():
    words = np.arange(65536, dtype="<u2")
    for bit in range(16):
        changed = words ^ (1 << bit)
        expanded = np.frombuffer(widen_bf16_bytes(changed.tobytes()), dtype="<u4")
        assert np.array_equal(expanded, changed.astype("<u4") << 16)
    with pytest.raises(ValueError, match="complete"):
        widen_bf16_bytes(b"\x00")


def fixture_storage():
    torch = pytest.importorskip("torch")
    from netsurf.bf16_fp32_storage import BF16StorageFP32Compute
    model = torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False)).eval()
    source = torch.tensor([[1., -0.5], [0., 0.25]], dtype=torch.bfloat16)
    with torch.no_grad():
        model[0].weight.copy_(source.float())
    return model, BF16StorageFP32Compute(model, {"0.weight": source})


def schedule(store, bits):
    return FaultSchedule(FaultModel("weight", "bf16_stored_fp32_compute", "persistent_trial",
                                   "forced_engineering", 0.), 31,
                         tuple(FaultEvent(None, BitIdentity("0.weight", 0, bit)) for bit in bits),
                         store.bit_count, 1)


@pytest.mark.parametrize("bits", [(15,), (7,), (0,), (7, 14), (14,), (14, 6)])
def test_mixed_storage_xor_reaches_inference_and_restores(bits):
    torch = pytest.importorskip("torch")
    model, store = fixture_storage()
    assert store.bit_count == 64  # NOT 128 FP32 computation bits
    none = ProtectionMask("none", frozenset(), 64, 0, {})
    before, runtime_before = store.hashes(), store.runtime_hashes()
    raw = store._bytes(store.parameters["0.weight"])[:2]
    expected = raw
    for bit in bits:
        expected = store.codec.xor(expected, 0, bit)
    x = torch.tensor([[1., 0.]])
    for _ in range(2):
        with store.injected(schedule(store, bits), none) as report:
            assert report.applied_events == len(bits)
            assert store._bytes(model[0].weight)[:4] == widen_bf16_bytes(expected)
            actual = model(x).detach()[0, 0].item()
            wanted = struct.unpack("<f", widen_bf16_bytes(expected))[0]
            assert (np.isnan(actual) and np.isnan(wanted)) or actual == wanted
        assert store.hashes() == before and store.runtime_hashes() == runtime_before
    protected = frozenset(e.bit for e in schedule(store, bits).events)
    mask = ProtectionMask("full", protected, 64, len(protected), {})
    with store.injected(schedule(store, bits), mask) as report:
        assert report.applied_events == 0 and report.suppressed_events == len(bits)
        assert model(x).detach()[0, 0].item() == 1.


def test_exception_invalid_fault_and_nested_transactions_restore():
    model, store = fixture_storage()
    none = ProtectionMask("none", frozenset(), 64, 0, {})
    before = store.hashes(), store.runtime_hashes()
    with pytest.raises(RuntimeError, match="evaluation failed"):
        with store.injected(schedule(store, (15,)), none):
            raise RuntimeError("evaluation failed")
    assert (store.hashes(), store.runtime_hashes()) == before
    with pytest.raises(ValueError, match="stored word"):
        with store.injected(schedule(store, (31,)), none):
            pass
    assert (store.hashes(), store.runtime_hashes()) == before
    with store.injected(schedule(store, (15,)), none):
        with pytest.raises(RuntimeError, match="nested"):
            with store.injected(schedule(store, (15,)), none):
                pass
    assert (store.hashes(), store.runtime_hashes()) == before


def test_mismatch_and_float_shadow_are_rejected():
    torch = pytest.importorskip("torch")
    from netsurf.bf16_fp32_storage import BF16StorageFP32Compute
    model, store = fixture_storage()
    source = store.parameters["0.weight"]
    with pytest.raises(ValueError, match="checkpoint BF16"):
        BF16StorageFP32Compute(model, {"0.weight": source.float()})
    with torch.no_grad():
        model[0].weight[0, 0] += 1e-4
    with pytest.raises(ValueError, match="exact BF16"):
        BF16StorageFP32Compute(model, {"0.weight": source})


def test_zero_word_faults_partial_suppression_and_counts():
    torch = pytest.importorskip("torch")
    model, store = fixture_storage()
    a, b = BitIdentity("0.weight", 2, 0), BitIdentity("0.weight", 2, 15)
    law = schedule(store, ()).model
    field = FaultSchedule(law, 1, (FaultEvent(None, a), FaultEvent(None, b)), 64, 1)
    mask = ProtectionMask("partial", frozenset((b,)), 64, 1, {})
    with store.injected(field, mask) as report:
        assert report.proposed_events == report.applied_events + report.suppressed_events == 2
        assert report.applied_events == report.suppressed_events == 1
        assert store._bytes(model[0].weight)[8:12] == b"\x00\x00\x01\x00"


def test_reference_nll_failures_are_not_survivor_perplexity():
    from netsurf.adapters.pythia_mixed import summarize_nll
    result = summarize_nll([1., 40., float('nan')], [1, 2, 3], [(0, 4, 1)])
    assert result.mean_nll is None and result.perplexity is None
    assert result.failed_tokens == result.clipped_finite_tokens == 1
    assert result.mean_capped_nll == pytest.approx(61 / 3)
    finite = summarize_nll([1., 40.], [1, 2], [(0, 3, 1)])
    assert finite.mean_nll == 20.5 and finite.mean_capped_nll == 15.5
    assert finite.token_nll_sha256 == hashlib.sha256(np.array([1., 40.], dtype='<f8').tobytes()).hexdigest()


def test_preparation_rejects_context_leakage_and_wrong_exposure():
    from netsurf.pythia_preflight import validate_preparation
    root = Path(__file__).resolve().parents[1]
    plan = json.loads((root / 'config/experiments/pythia_bf16_storage_fp32_prepare_v1.json').read_text())
    inventory = {'eligible_bits': 1977090048, 'tensors': [
        {'name': plan['fault_support']['tensor'], 'words': 1769472}]}
    result = validate_preparation(plan, inventory)
    assert result['ber'] == [1/128, 4/128, 16/128]
    assert result['total_forward_windows'] == 899
    assert result['independent_final_fields'] == 48
    plan['cohorts']['final']['window_indices'] = [1]  # Different index, overlapping context.
    with pytest.raises(ValueError, match='context overlap'):
        validate_preparation(plan, inventory)
    plan['cohorts']['final']['window_indices'] = [4, 5]
    plan['fault_support']['eligible_bits'] = 256
    with pytest.raises(ValueError, match='bit count'):
        validate_preparation(plan, inventory)


def test_mixed_ready_requires_separate_identity_and_hash_bound_reference(tmp_path):
    from netsurf.pythia_assets import TARGET, CONTRACT, load_pythia_asset
    from netsurf.transformer_assets import AssetBlocked
    def file(name, contents):
        path = tmp_path / name
        path.write_text(contents)
        return {'path': name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    files = [file(name, name) for name in ('model.safetensors', 'config.json',
        'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json')]
    data_files = [file('data.parquet', 'fixture data')]
    env = file('env.lock', 'pinned environment')
    file('python', 'entrypoint')
    pin = {'repository': 'fixture/derived', 'commit': 'a'*40,
           'files': {r['path']: r['sha256'] for r in files},
           'derivation': {'source': 'fixture float32 to bf16 RNE'},
           'dataset': {'source': 'Salesforce/wikitext', 'revision': 'b'*40,
                       'files': {r['path']: r['sha256'] for r in data_files}}}
    pin_path = tmp_path / 'pin.json'; pin_path.write_text(json.dumps(pin))
    report = {'computation_dtype': 'float32', 'stored_dtype': {'BF16': 1},
        'repeat_bitwise_identical': True, 'protocol': 'kona-assets/lm-sliding-window/1',
        'model': {'commit': pin['commit'], 'files': files},
        'windowing': {'context': 1024, 'stride': 512, 'bos': 'none'},
        'per_target_nll_sha256': 'c'*64, 'metrics': {'mean_nll_nats': 3., 'perplexity': 20.},
        'dataset': {'split': 'test', 'token_ids_sha256': 'd'*64, 'text_sha256': 'e'*64},
        'provenance': {'cuda_available': False, 'deterministic_algorithms': True, 'attention': 'eager',
                       'threads': 32, 'python': '3.11.18', 'torch': '2.9.1+cpu', 'transformers': '4.57.1'}}
    ref = file('reference.json', json.dumps(report))
    source = {'target_id': TARGET, 'status': 'READY',
        'execution': {'contract': CONTRACT, 'stored_precision': 'BF16', 'compute_precision': 'FP32'},
        'model': {'repository': pin['repository'], 'commit': pin['commit'], 'path': str(tmp_path), 'files': files},
        'tokenizer': files[2:], 'derivation': pin['derivation'],
        'dataset': {'source': 'Salesforce/wikitext', 'revision': 'b'*40, 'config': 'wikitext-2-raw-v1',
                    'path': str(tmp_path), 'files': data_files},
        'reference_results': {'smoke_report': str(tmp_path/'reference.json'), 'smoke_report_sha256': ref['sha256'],
                              'gate': {'accepted_quality': True}, 'metrics': report['metrics']},
        'interpreter': str(tmp_path/'python'),
        'environment': {'interpreter': str(tmp_path/'python'), 'requirements_lock': str(tmp_path/'env.lock'),
                        'requirements_lock_sha256': env['sha256'], 'device': 'cpu', 'attention': 'eager',
                        'python': '3.11.18', 'torch': '2.9.1+cpu', 'packages': {'transformers': '4.57.1'}}}
    manifest = tmp_path/'readiness.json'
    def save():
        manifest.write_text(json.dumps({'schema': 'kona-assets/netsurf-asset-readiness/1', 'targets': [source]}))
    save()
    assert load_pythia_asset(manifest, pin_path).backend['threads'] == 32
    source['target_id'] = 'pythia160m_bf16'; save()
    with pytest.raises(AssetBlocked, match='distinct finalized'):
        load_pythia_asset(manifest, pin_path)
    source['target_id'] = TARGET; source['execution']['compute_precision'] = 'BF16'; save()
    with pytest.raises(AssetBlocked, match='explicitly identify'):
        load_pythia_asset(manifest, pin_path)
    source['execution']['compute_precision'] = 'FP32'; save()
    (tmp_path/'reference.json').write_text('{}')
    with pytest.raises(AssetBlocked, match='hash-bound'):
        load_pythia_asset(manifest, pin_path)
