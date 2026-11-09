"""Bounded live-tensor XOR/restoration check with the pinned W8A8 interpreter."""

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

from netsurf.w8a8_assets import KONA_ROOT
from netsurf.w8a8_dense_ber_pilot import injected_packed_unprotected, packed_field
from netsurf.w8a8_dense_protection import Protection, injected_packed, boundary_protections
from netsurf.w8a8_faults import MSBMask


def main() -> None:
    sys.path.insert(0, str(KONA_ROOT / "src"))
    from kona_assets import int8_torch as execution
    modules = {
        "first": SimpleNamespace(qweight=torch.tensor([-128, 1], dtype=torch.int8)),
        "second": SimpleNamespace(qweight=torch.tensor([-1, 127], dtype=torch.int8)),
    }
    original = {name: module.qweight.clone() for name, module in modules.items()}
    adapter = SimpleNamespace(
        modules=modules,
        baseline_hashes={name: execution.tensor_sha256(module.qweight) for name, module in modules.items()},
        universe=SimpleNamespace(bits=32, codes=4, entries=[{"module": name} for name in modules]),
    )
    with tempfile.TemporaryDirectory(prefix="netsurf-dense-xor-") as temporary:
        path = Path(temporary) / "field.u8"
        field = packed_field(path, 32, 0.7, 919, chunk_codes=1)
        mask = np.frombuffer(path.read_bytes(), dtype=np.uint8)
        with injected_packed_unprotected(adapter, path, field) as counts:
            assert counts["proposed"] == counts["applied"] == field["proposed"]
            actual = np.concatenate([module.qweight.view(torch.uint8).numpy() for module in modules.values()])
            expected = np.bitwise_xor(np.concatenate([
                tensor.view(torch.uint8).numpy() for tensor in original.values()]), mask)
            assert np.array_equal(actual, expected)
        assert all(torch.equal(modules[name].qweight, original[name]) for name in modules)
        proposals = np.memmap(path, dtype=np.uint8, mode="r")
        for budget in (1, 4, 5, 8, 16, 24, 31):
            ranking = MSBMask(32, budget)
            protection = Protection("msb", 32, budget, ranking.digest)
            with injected_packed(adapter, proposals, field, protection) as counts:
                expected = np.bitwise_xor(np.concatenate([
                    tensor.view(torch.uint8).numpy() for tensor in original.values()]), mask)
                for bit in range(32):
                    if ranking.contains(bit):
                        expected[bit//8] ^= np.uint8(mask[bit//8] & (1 << (bit % 8)))
                actual = np.concatenate([module.qweight.view(torch.uint8).numpy()
                                         for module in modules.values()])
                assert np.array_equal(actual, expected)
                assert counts["proposed"] == counts["applied"] + counts["suppressed"]
            assert all(torch.equal(modules[name].qweight, original[name]) for name in modules)
        for protection in boundary_protections(32).values():
            with injected_packed(adapter, proposals, field, protection) as counts:
                assert counts["applied"] == (0 if protection.kind == "full" else field["proposed"])
            assert all(torch.equal(modules[name].qweight, original[name]) for name in modules)
        try:
            with injected_packed_unprotected(adapter, path, field):
                raise ArithmeticError("intentional bounded failure")
        except ArithmeticError:
            pass
        assert all(torch.equal(modules[name].qweight, original[name]) for name in modules)
    print("PINNED_DENSE_XOR_RESTORATION_PASS")


if __name__ == "__main__":
    main()
