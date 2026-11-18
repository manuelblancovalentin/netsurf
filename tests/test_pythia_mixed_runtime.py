"""Run with stdlib unittest in the pinned environment; pytest is not required."""
import unittest
import importlib.util
import math


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional pinned PyTorch backend")
class MixedRuntimeTests(unittest.TestCase):
    def setUp(self):
        import torch
        from netsurf.bf16_fp32_storage import BF16StorageFP32Compute
        self.torch = torch
        self.model = torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False)).eval()
        source = torch.tensor([[1., -0.5], [0., 0.25]], dtype=torch.bfloat16)
        with torch.no_grad():
            self.model[0].weight.copy_(source.float())
        self.store = BF16StorageFP32Compute(self.model, {"0.weight": source})

    def field(self, bits):
        from netsurf.faults import FaultModel, FaultSchedule, FaultEvent
        from netsurf.identities import BitIdentity
        from netsurf.protection import ProtectionMask
        field = FaultSchedule(FaultModel("weight", "bf16_stored_fp32_compute", "persistent_trial",
            "forced_engineering", 0.), 7, tuple(FaultEvent(None, BitIdentity("0.weight", 0, b)) for b in bits),
            64, 1)
        return field, ProtectionMask("none", frozenset(), 64, 0, {})

    def test_sign_exponent_significand_multibit_inf_nan_and_replay(self):
        from netsurf.bf16_fp32_storage import widen_bf16_bytes
        from netsurf.protection import ProtectionMask
        before = self.store.hashes(), self.store.runtime_hashes()
        x = self.torch.tensor([[1., 0.]])
        for bits in [(15,), (7,), (0,), (7, 14), (14,), (14, 6)]:
            field, none = self.field(bits)
            expected = b"\x80\x3f"
            for bit in bits:
                expected = self.store.codec.xor(expected, 0, bit)
            for _ in range(2):
                with self.store.injected(field, none) as report:
                    self.assertEqual(report.applied_events, len(bits))
                    self.assertEqual(self.store._bytes(self.model[0].weight)[:4], widen_bf16_bytes(expected))
                    result = self.model(x)[0, 0].item()
                    wanted = self.store.codec.value(expected)
                    self.assertTrue((math.isnan(result) and math.isnan(wanted)) or result == wanted)
                self.assertEqual((self.store.hashes(), self.store.runtime_hashes()), before)
            mask = ProtectionMask("suppress_all_events", frozenset(e.bit for e in field.events), 64, len(bits), {})
            with self.store.injected(field, mask) as report:
                self.assertEqual(report.applied_events, 0)
                self.assertEqual(self.model(x)[0, 0].item(), 1.)

    def test_exception_restoration_and_ineligible_fp32_bit_rejection(self):
        before = self.store.hashes(), self.store.runtime_hashes()
        field, none = self.field((15,))
        with self.assertRaisesRegex(RuntimeError, "deliberate"):
            with self.store.injected(field, none):
                raise RuntimeError("deliberate evaluation failure")
        self.assertEqual((self.store.hashes(), self.store.runtime_hashes()), before)
        bad, none = self.field((31,))
        with self.assertRaisesRegex(ValueError, "stored word"):
            with self.store.injected(bad, none):
                pass
        self.assertEqual((self.store.hashes(), self.store.runtime_hashes()), before)

    def test_bf16_not_fp32_population_and_off_lattice_rejection(self):
        from netsurf.bf16_fp32_storage import BF16StorageFP32Compute
        self.assertEqual(self.store.bit_count, 64)
        self.assertEqual(self.store.bit_at(63).bit_index, 15)
        with self.torch.no_grad():
            self.model[0].weight[0, 0] += 1e-4
        with self.assertRaisesRegex(ValueError, "exact BF16"):
            BF16StorageFP32Compute(self.model, self.store.parameters)

    def test_lm_reference_reduction_and_persistent_fault(self):
        from types import SimpleNamespace
        from netsurf.adapters.pythia_mixed import PythiaBF16StorageAdapter
        from netsurf.bf16_fp32_storage import BF16StorageFP32Compute
        from netsurf.pythia_assets import TARGET
        torch = self.torch
        class TinyLM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = torch.nn.Linear(4, 4, bias=False)
                with torch.no_grad():
                    self.projection.weight.copy_(torch.eye(4))
            def forward(self, input_ids, use_cache=False):
                return SimpleNamespace(logits=self.projection(torch.nn.functional.one_hot(input_ids, 4).float()))
        adapter = PythiaBF16StorageAdapter(SimpleNamespace(target=TARGET))
        adapter.model = TinyLM().eval()
        adapter.tokenizer = object()
        adapter.storage = BF16StorageFP32Compute(adapter.model,
            {"projection.weight": adapter.model.projection.weight.detach().to(torch.bfloat16)})
        tokens = (0, 1, 2, 3) * 384
        first = adapter.evaluate_windows(tokens, window_indices=[0, 1])
        self.assertEqual(first, adapter.evaluate_windows(tokens, window_indices=[0, 1]))
        self.assertEqual(first.scored_token_positions, tuple(range(1, 1536)))
        self.assertEqual(first.failed_tokens, 0)
        self.assertAlmostEqual(first.perplexity, math.exp(first.mean_nll))
        # No short-context substitution: the second full window retains the
        # original 512-token overlap even when evaluated by itself.
        second = adapter.evaluate_windows(tokens, window_indices=[1])
        self.assertEqual(second.token_nll, first.token_nll[1023:])


if __name__ == "__main__":
    unittest.main()
