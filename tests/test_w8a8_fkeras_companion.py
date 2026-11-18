"""Tiny continuous decoded-code companion fixture, no model asset required."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from scripts.preflight_w8a8_fkeras_full_v1 import prepare_companion


class FakeQLinear(torch.nn.Module):
    def __init__(self, codes, scales):
        super().__init__()
        self.register_buffer("qweight", torch.tensor(codes, dtype=torch.int8))
        self.register_buffer("scale", torch.tensor(scales, dtype=torch.float32))
        self.in_features = len(codes[0])
        self.out_features = len(codes)
        self.bias = None

    def forward(self, values):
        return torch.nn.functional.linear(values, self.qweight.float() * self.scale[:, None])


def test_every_eligible_qweight_maps_to_exact_decoded_linear_and_hvp():
    model = torch.nn.Sequential(FakeQLinear([[1, -2], [3, 4]], [.25, .5]),
                                torch.nn.Tanh(), FakeQLinear([[5, -1]], [.125]))
    entries = ({"module": "0", "shape": [2, 2]}, {"module": "2", "shape": [1, 2]})
    adapter = SimpleNamespace(model=model, modules={"0": model[0], "2": model[2]},
                              universe=SimpleNamespace(entries=entries, bits=48))
    original = model(torch.tensor([[.5, -.25]])).detach()
    mapping, weights = prepare_companion(adapter)
    assert mapping == (("0", 0, 4), ("2", 4, 2))
    assert len(weights) == 2 and sum(w.numel() for w in weights) == 6
    assert adapter.modules is None
    assert isinstance(model[0], torch.nn.Linear) and isinstance(model[2], torch.nn.Linear)
    torch.testing.assert_close(model(torch.tensor([[.5, -.25]])).detach(), original)
    torch.testing.assert_close(model[0].weight.detach(), torch.tensor([[.25, -.5], [1.5, 2.]]))
    x = torch.tensor([[.5, -.25]])
    loss = model(x).square().sum()
    grads = torch.autograd.grad(loss, weights, create_graph=True)
    direction = tuple(torch.ones_like(w) for w in weights)
    products = torch.autograd.grad(grads, weights, grad_outputs=direction)
    assert all(torch.isfinite(product).all() for product in products)
    assert sum(p.numel() for p in products) == 6
