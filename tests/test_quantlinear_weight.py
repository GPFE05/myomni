"""Test QuantLinear weight type"""
import torch
import torch.nn as nn
import sys
sys.path.insert(0, r'd:\1intel\myomni')
from quantize.int_linear import QuantLinear

# Create original Linear
linear = nn.Linear(10, 20)
print(f"Original weight type: {type(linear.weight)}")
print(f"Original weight is Parameter: {isinstance(linear.weight, nn.Parameter)}")

# Create QuantLinear
ql = QuantLinear(linear, {}, {})
print(f"\nQuantLinear weight type: {type(ql.weight)}")
print(f"QuantLinear weight is Parameter: {isinstance(ql.weight, nn.Parameter)}")
print(f"'weight' in _buffers: {'weight' in ql._buffers}")
print(f"'weight' in _parameters: {'weight' in ql._parameters}")

# Test assigning temp_weight
print("\n--- Testing temp_weight assignment (original way - will fail) ---")
try:
    ql.temp_weight = ql.weight
    print(f"First assignment OK. temp_weight type: {type(ql.temp_weight)}")
    print(f"'temp_weight' in _parameters: {'temp_weight' in ql._parameters}")
    
    # Now try to assign a regular tensor
    new_tensor = torch.randn(20, 10)
    ql.temp_weight = new_tensor
    print(f"Second assignment OK. temp_weight type: {type(ql.temp_weight)}")
except Exception as e:
    print(f"ERROR (expected): {type(e).__name__}: {e}")

# Test the fixed way
print("\n--- Testing temp_weight assignment (fixed way - using .data) ---")
ql2 = QuantLinear(nn.Linear(10, 20), {}, {})
try:
    ql2.temp_weight = ql2.weight.data  # Use .data to get plain tensor
    print(f"First assignment OK. temp_weight type: {type(ql2.temp_weight)}")
    print(f"'temp_weight' in _parameters: {'temp_weight' in ql2._parameters}")
    
    # Now try to assign a regular tensor
    new_tensor = torch.randn(20, 10)
    ql2.temp_weight = new_tensor
    print(f"Second assignment OK. temp_weight type: {type(ql2.temp_weight)}")
    print("SUCCESS: Fix works!")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
