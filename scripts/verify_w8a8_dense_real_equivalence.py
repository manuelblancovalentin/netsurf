"""Bounded real-DistilBERT equivalence of sparse and packed protected XOR."""

import hashlib

import numpy as np

from netsurf.w8a8_assets import load_ready_w8a8
from netsurf.w8a8_dense_protection import Protection, injected_packed
from netsurf.w8a8_faults import MSBMask, W8A8FaultAdapter, W8A8Field
from netsurf.w8a8_rank import PackedMask
from netsurf.w8a8_wide_ber_campaign import _source, _policy_records


def main() -> None:
    manifest = '/projects/kona-assets/manuelbv/NETSURF_ASSET_READINESS.json'
    asset = load_ready_w8a8(manifest, 'distilbert_sst2_int8')
    adapter = W8A8FaultAdapter(asset)
    adapter.load()
    rows = adapter.sst2_cohorts()['engineering'][:8]
    evaluate = lambda: adapter.evaluate_sst2(rows)
    clean = evaluate()
    n = adapter.universe.bits
    # Two flips in one stored word, protected and unprotected bit positions,
    # and coordinates near the beginning and end of the eligible population.
    ordinals = (0, 7, 8, 9, 15, n//2, n//2+7, n-8, n-1)
    packed = np.zeros(n//8, dtype=np.uint8)
    for ordinal in ordinals:
        packed[ordinal//8] ^= np.uint8(1 << (ordinal % 8))
    receipt = {'universe_bits': n, 'proposed': len(ordinals),
               'sha256': hashlib.sha256(packed.tobytes()).hexdigest()}
    for budget in (0, n//16, n//8, n//4, n):
        sparse_mask = MSBMask(n, budget)
        packed_mask = Protection('msb', n, budget, sparse_mask.digest)
        with adapter.injected(W8A8Field(1, 0.0, n, ordinals), sparse_mask) as sparse_events:
            sparse_risk = evaluate()
        with injected_packed(adapter, packed, receipt, packed_mask) as packed_events:
            packed_risk = evaluate()
        assert sparse_risk == packed_risk
        assert packed_events['proposed'] == sparse_events['proposed']
        assert packed_events['applied'] == sparse_events['applied']
        assert packed_events['suppressed'] == sparse_events['suppressed']
        assert evaluate() == clean
    _, source_run, source_plan, seal = _source('distilbert_sst2_int8')
    records = _policy_records(n, source_run, source_plan, seal)
    selected = next(item for item in records if item['name'] == f'per_example_absolute_b{n//8}')
    original = PackedMask(__import__('pathlib').Path(selected['path']), n, n//8,
                          selected['file_sha256'])
    same_mask = Protection('packed', n, n//8, selected['mask_digest'],
                           __import__('pathlib').Path(selected['path']),
                           selected['file_sha256'])
    protected_bytes = np.memmap(selected['path'], dtype=np.uint8, mode='r')
    with adapter.injected(W8A8Field(1, 0.0, n, ordinals), original) as sparse_events:
        sparse_risk = evaluate()
    with injected_packed(adapter, packed, receipt, same_mask, protected_bytes) as packed_events:
        packed_risk = evaluate()
    assert sparse_risk == packed_risk
    assert packed_events['applied'] == sparse_events['applied']
    assert evaluate() == clean
    print('REAL_DISTILBERT_SPARSE_PACKED_PROTECTED_EQUIVALENCE_PASS')


if __name__ == '__main__':
    main()
