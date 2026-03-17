import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantize.utils import (
    forward_with_module_input,
    set_stage_trainable_params,
    assert_tensors_on_same_device,
)


class DummyGateLayer(nn.Module):
    def __init__(self, hidden_size=8):
        super().__init__()
        self.self_attn = nn.Linear(hidden_size, hidden_size)
        self.mlp = nn.Module()
        self.mlp.gate = nn.Linear(hidden_size, 4)
        self.shared_expert_gate = nn.Linear(hidden_size, 2)
        self.smooth_scale = nn.Parameter(torch.ones(hidden_size))
        self.bound_factor = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x, **kwargs):
        attn_out = self.self_attn(x)
        _ = self.mlp.gate(attn_out)
        return attn_out


def test_forward_with_module_input_captures_post_attn_residual_stream():
    layer = DummyGateLayer()
    x = torch.randn(2, 4, 8)

    out, norm_in = forward_with_module_input(layer, x, module_path="mlp.gate")

    assert out.shape == x.shape
    assert norm_in is not None
    assert norm_in.shape == x.shape


def test_stage_trainable_params_ri_and_mi():
    layer = DummyGateLayer()

    ri_named = set_stage_trainable_params(layer, stage="ri")
    ri_names = {name for name, _ in ri_named}
    assert "mlp.gate.weight" in ri_names
    assert all("smooth" not in n for n in ri_names)

    mi_named = set_stage_trainable_params(
        layer,
        stage="mi",
        use_shift=True,
        train_shared_gate=True,
        train_gate_lora=False,
    )
    mi_names = {name for name, _ in mi_named}
    assert any("smooth" in n for n in mi_names)
    assert any("bound_factor" in n for n in mi_names)
    assert any("shared_expert_gate" in n for n in mi_names)


def test_assert_tensors_on_same_device_cpu_pass():
    x = torch.randn(2, 2)
    y = torch.randn(2, 2)
    assert_tensors_on_same_device({"x": x, "y": y}, context="cpu-pass")
