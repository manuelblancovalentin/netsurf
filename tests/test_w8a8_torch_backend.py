"""Standard-library fixture; run with the pinned Kona torch interpreter."""

from types import SimpleNamespace
import sys
from tempfile import TemporaryDirectory
from pathlib import Path
import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    sys.path.insert(0, "/projects/kona-assets/manuelbv/src")
    from kona_assets.int8_torch import QLinear, tensor_sha256
    from netsurf.w8a8_faults import W8A8FaultAdapter, W8A8Field, FixedMask
    from netsurf.w8a8_surrogate import build_surrogate_moments
    from netsurf.w8a8_rank import gradient_score_chunks


@unittest.skipIf(torch is None, "pinned PyTorch interpreter required")
class W8A8TorchTests(unittest.TestCase):
    def test_live_int8_xor_persists_and_restores_even_after_exception(self):
        first = QLinear(2, 2, False)
        second = QLinear(1, 2, False)
        with torch.no_grad():
            first.qweight.copy_(torch.tensor([[1, 0], [100, 0]], dtype=torch.int8))
            first.scale.copy_(torch.tensor([.1, 1.], dtype=torch.float32))
            second.qweight.copy_(torch.tensor([[0, 1]], dtype=torch.int8))
        asset = SimpleNamespace(quantization={"tensors": [
            {"module": "first", "shape": [2, 2], "qweight_sha256": tensor_sha256(first.qweight),
             "scale_sha256": tensor_sha256(first.scale)},
            {"module": "second", "shape": [1, 2], "qweight_sha256": tensor_sha256(second.qweight),
             "scale_sha256": tensor_sha256(second.scale)}]}, eligible_bits=48)
        adapter = W8A8FaultAdapter(asset)
        adapter.model = object()
        adapter.modules = {"first": first, "second": second}
        adapter.baseline_hashes = {name: tensor_sha256(module.qweight) for name, module in adapter.modules.items()}
        x = torch.tensor([[1., 0.]])
        def run():
            intermediate = first(x)
            scale = (intermediate.abs().amax(dim=1, keepdim=True)/127).clamp_min(1e-12)
            codes = torch.round(intermediate/scale).clamp(-127,127).to(torch.int8)
            return intermediate, codes, second(intermediate)
        clean_middle, clean_codes, clean_final = run()
        ordinal = adapter.universe.ordinal("first", 0, 0)
        field = W8A8Field(7, 0., 48, (ordinal,), "forced_engineering")
        unprotected = FixedMask(48, frozenset())
        with adapter.injected(field, unprotected) as report:
            damaged_middle, damaged_codes, damaged_final = run()
            self.assertEqual((report["applied"], report["suppressed"]), (1, 0))
            self.assertFalse(torch.equal(clean_middle, damaged_middle))
            self.assertTrue(torch.equal(clean_codes, damaged_codes))
            self.assertTrue(torch.equal(clean_final, damaged_final))
        self.assertEqual(tensor_sha256(first.qweight), adapter.baseline_hashes["first"])
        with self.assertRaisesRegex(RuntimeError, "injected"):
            with adapter.injected(field, unprotected):
                raise RuntimeError("injected exception")
        self.assertEqual(tensor_sha256(first.qweight), adapter.baseline_hashes["first"])
        with adapter.injected(field, FixedMask(48, frozenset({ordinal}))) as report:
            self.assertEqual(report["applied"], 0)
            self.assertTrue(torch.equal(run()[2], clean_final))

    def test_exact_decoded_code_surrogate_moments(self):
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.first = QLinear(2, 2, False)
                self.classifier = QLinear(2, 2, False)
                with torch.no_grad():
                    self.first.qweight.copy_(torch.tensor([[4, 1], [-2, 3]], dtype=torch.int8))
                    self.classifier.qweight.copy_(torch.tensor([[3, -1], [-2, 2]], dtype=torch.int8))
            def forward(self, input_ids):
                return SimpleNamespace(logits=self.classifier(self.first(input_ids.float())))
        model = Tiny()
        modules = {"first": model.first, "classifier": model.classifier}
        asset = SimpleNamespace(target="distilbert_sst2_int8", record_sha256="a"*64,
            quantization={"tensors": [{"module": n, "shape": [2,2],
                "qweight_sha256": tensor_sha256(m.qweight), "scale_sha256": tensor_sha256(m.scale)}
                for n,m in modules.items()]}, eligible_bits=64)
        adapter = W8A8FaultAdapter(asset)
        adapter.model = model
        adapter.modules = modules
        adapter.tokenizer = lambda texts, **kwargs: {"input_ids": torch.tensor([[1.,2.]] if texts[0]=="a" else [[2.,1.]])}
        adapter.sst2_cohorts = lambda: {"calibration": (SimpleNamespace(text="a",label=0,source_index=1),
                                                      SimpleNamespace(text="b",label=1,source_index=2))}
        with TemporaryDirectory() as d:
            moments = build_surrogate_moments(adapter,Path(d)/"moments",sst2_examples=2)
            self.assertEqual(moments["examples"],2)
            self.assertEqual(set(moments["tensors"]),set(modules))
            scores = list(gradient_score_chunks(modules,adapter.universe,Path(d)/"moments",
                                                 "per_example_positive"))
            self.assertEqual(sum(v.size for _,v in scores),64)


if __name__ == "__main__":
    unittest.main()
