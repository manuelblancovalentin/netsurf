"""Synthetic BF16-source/FP32-computation fault contract."""

from copy import deepcopy
from types import SimpleNamespace
import unittest

try:
    import torch
except ImportError:
    torch = None

from netsurf.adapters.transformers_pinned import PinnedTransformerAdapter
from netsurf.faults import FaultEvent, FaultModel, FaultSchedule
from netsurf.identities import BitIdentity
from netsurf.protection import ProtectionMask
from netsurf.torch_storage import TorchStoredLinearWeights


@unittest.skipUnless(torch is not None, "PyTorch not installed in the NetSuRF test environment")
class Bf16StorageContractTest(unittest.TestCase):
  def test_bf16_stored_fault_is_widened_consumed_and_restored(self):
    source = torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False)).to(torch.bfloat16).eval()
    with torch.no_grad():
        source[0].weight.copy_(torch.tensor([[1.0, 0.5], [-0.25, 0.75]], dtype=torch.bfloat16))
    compute = deepcopy(source).to(torch.float32).eval()
    bit = BitIdentity("0.weight", 0, 15)
    adapter = PinnedTransformerAdapter(SimpleNamespace(precision="BF16",
        computation_precision="FP32"))
    adapter.storage_model = source
    adapter.model = compute
    adapter.tokenizer = object()
    adapter.storage = TorchStoredLinearWeights(source, "BF16", eligible_bits=(bit,))
    adapter._sync_computation_model()
    before = adapter.storage.hashes()
    input_vector = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    clean = compute(input_vector).detach().clone()
    law = FaultModel("weight", "synthetic_bf16_source", "persistent_trial",
                     "forced_engineering", 0.0)
    schedule = FaultSchedule(law, 41, (FaultEvent(None, bit),), 1, 1)
    unprotected = ProtectionMask("none", frozenset(), 1, 0, {})
    with adapter.injected(schedule, unprotected) as report:
        assert report.applied_events == 1
        assert adapter.storage.hashes() != before
        assert compute[0].weight[0, 0].item() == -1.0
        assert not torch.equal(compute(input_vector), clean)
    assert adapter.storage.hashes() == before
    assert torch.equal(compute(input_vector), clean)
