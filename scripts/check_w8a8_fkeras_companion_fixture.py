#!/usr/bin/env python3
"""No-pytest pinned-PyTorch fixture for every-qweight companion mapping."""

from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from preflight_w8a8_fkeras_full_v1 import prepare_companion


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


def main() -> None:
    model = torch.nn.Sequential(FakeQLinear([[1, -2], [3, 4]], [.25, .5]),
                                torch.nn.Tanh(), FakeQLinear([[5, -1]], [.125]))
    entries = ({"module": "0", "shape": [2, 2]}, {"module": "2", "shape": [1, 2]})
    adapter = SimpleNamespace(model=model, modules={"0": model[0], "2": model[2]},
                              universe=SimpleNamespace(entries=entries, bits=48))
    x = torch.tensor([[.5, -.25]])
    original = model(x).detach()
    mapping, weights = prepare_companion(adapter)
    assert mapping == (("0", 0, 4), ("2", 4, 2))
    torch.testing.assert_close(model(x).detach(), original)
    torch.testing.assert_close(model[0].weight.detach(), torch.tensor([[.25, -.5], [1.5, 2.]]))
    loss = model(x).square().sum()
    gradients = torch.autograd.grad(loss, weights, create_graph=True)
    products = torch.autograd.grad(gradients, weights,
                                   grad_outputs=tuple(torch.ones_like(w) for w in weights))
    assert len(products) == 2 and all(torch.isfinite(h).all() for h in products)
    print("PASS: full synthetic qweight mapping, decoded FP32 forward and HVP")


if __name__ == "__main__":
    main()
