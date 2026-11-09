import hashlib

import numpy as np
import pytest

from netsurf.w8a8_dense_protection import (Protection, msb_protection_bytes,
    protection_chunk, verify_protection)
from netsurf.w8a8_faults import MSBMask
from netsurf.w8a8_rank import PackedMask


@pytest.mark.parametrize("codes", (1, 2, 7, 17))
def test_msb_packed_bytes_equal_canonical_bit_rank(codes):
    for budget in range(8*codes + 1):
        result = np.concatenate([msb_protection_bytes(start, min(3, codes-start), codes, budget)
                                 for start in range(0, codes, 3)])
        original = MSBMask(8*codes, budget)
        actual = {8*word+bit for word in range(codes) for bit in range(8)
                  if result[word] & (1 << bit)}
        expected = {ordinal for ordinal in range(8*codes) if original.contains(ordinal)}
        assert actual == expected
        assert len(actual) == budget


def test_packed_protection_exact_cardinality_and_hash(tmp_path):
    path = tmp_path / "mask.bits"
    path.write_bytes(bytes([0b10000001, 0b01000000, 0]))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    mask = PackedMask(path, 24, 3, digest)
    protection = Protection("packed", 24, 3, mask.digest, path, digest)
    assert verify_protection(protection)["budget"] == 3
    packed = np.memmap(path, dtype=np.uint8, mode="r")
    assert protection_chunk(protection, 0, 3, packed).tolist() == [129, 64, 0]
    with pytest.raises(ValueError, match="cardinality"):
        verify_protection(Protection("packed", 24, 4,
            PackedMask(path, 24, 4, digest).digest, path, digest))
