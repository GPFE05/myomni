import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantize.omniquant import (
    stage_build_units,
    stage_should_quantize_linear,
    stage_is_router_gate,
)


def test_stage_build_units():
    units = stage_build_units(4)
    assert units == [("A0", 0), ("Ai", 0), ("Ai", 1), ("Ai", 2), ("Z", 3)]


def test_stage_quantize_rules():
    # A0: only attention
    assert stage_should_quantize_linear("A0", "self_attn.q_proj") is True
    assert stage_should_quantize_linear("A0", "mlp.experts.0.up_proj") is False

    # Ai: moe only
    assert stage_should_quantize_linear("Ai", "self_attn.q_proj") is False
    assert stage_should_quantize_linear("Ai", "mlp.experts.0.up_proj") is True

    # Z: moe only
    assert stage_should_quantize_linear("Z", "self_attn.o_proj") is False
    assert stage_should_quantize_linear("Z", "mlp.gate") is True


def test_router_gate_name_match():
    assert stage_is_router_gate("mlp.gate") is True
    assert stage_is_router_gate("gate") is True
    assert stage_is_router_gate("mlp.experts.0.gate_proj") is False
