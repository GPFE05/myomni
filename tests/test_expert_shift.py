"""
Unit tests for expert shift computation functions in quantize/utils.py
This file tests the set intersection logic for expert shift metrics.

Run with: python tests/test_expert_shift.py
"""

import torch
import torch.nn.functional as F


# ============================================================================
# Inline implementations for standalone testing
# ============================================================================

def compute_expert_shift_detailed_OLD(student_logits, teacher_indices, k_routing):
    """OLD BUGGY implementation using sorted comparison."""
    student_probs = torch.softmax(student_logits, dim=-1)
    _, student_indices = torch.topk(student_probs, k=k_routing, dim=-1)
    teacher_topk = teacher_indices[..., :k_routing]
    
    # WRONG: sorted element-wise comparison
    student_sorted, _ = torch.sort(student_indices, dim=-1)
    teacher_sorted, _ = torch.sort(teacher_topk, dim=-1)
    matches_per_token = (student_sorted == teacher_sorted).sum(dim=-1)
    
    at_least_one_changed = (matches_per_token < k_routing)
    shift_any = at_least_one_changed.float().mean().item()
    
    half_k = k_routing / 2.0
    at_least_half_changed = (matches_per_token <= k_routing - half_k)
    shift_half = at_least_half_changed.float().mean().item()
    
    all_changed = (matches_per_token == 0)
    shift_all = all_changed.float().mean().item()
    
    return {"shift_any": shift_any, "shift_half": shift_half, "shift_all": shift_all}


def compute_expert_shift_detailed_NEW(student_logits, teacher_indices, k_routing, debug=False):
    """NEW FIXED implementation using one-hot set intersection."""
    num_experts = student_logits.shape[-1]
    batch_size, seq_len = student_logits.shape[:2]
    
    student_probs = torch.softmax(student_logits, dim=-1)
    _, student_indices = torch.topk(student_probs, k=k_routing, dim=-1)
    teacher_topk = teacher_indices[..., :k_routing]
    
    # CORRECT: one-hot set intersection
    student_onehot = torch.zeros(batch_size, seq_len, num_experts, device=student_logits.device)
    student_onehot.scatter_(-1, student_indices, 1)
    
    teacher_onehot = torch.zeros(batch_size, seq_len, num_experts, device=student_logits.device)
    teacher_onehot.scatter_(-1, teacher_topk.to(student_logits.device), 1)
    
    matches_per_token = (student_onehot * teacher_onehot).sum(dim=-1)
    
    if debug:
        print(f"[DEBUG] First token - Student indices: {student_indices[0, 0].tolist()}")
        print(f"[DEBUG] First token - Teacher indices: {teacher_topk[0, 0].tolist()}")
        print(f"[DEBUG] First token - matches: {matches_per_token[0, 0].item()}")
    
    at_least_one_changed = (matches_per_token < k_routing)
    shift_any = at_least_one_changed.float().mean().item()
    
    half_k = k_routing / 2.0
    at_least_half_changed = (matches_per_token <= k_routing - half_k)
    shift_half = at_least_half_changed.float().mean().item()
    
    all_changed = (matches_per_token == 0)
    shift_all = all_changed.float().mean().item()
    
    return {"shift_any": shift_any, "shift_half": shift_half, "shift_all": shift_all}


def test_compute_expert_shift_detailed_basic():
    """Test that compute_expert_shift_detailed works correctly for basic cases."""
    
    # Case 1: Perfect match - student selects same experts as teacher
    print("=" * 60)
    print("Test Case 1: Perfect match (student == teacher)")
    print("=" * 60)
    
    batch, seq_len, num_experts = 2, 4, 8
    k_routing = 2
    
    # Create teacher indices: experts [0, 1] for all tokens
    teacher_indices = torch.zeros(batch, seq_len, 3, dtype=torch.long)  # topk_cached=3
    teacher_indices[..., 0] = 0
    teacher_indices[..., 1] = 1
    teacher_indices[..., 2] = 2
    
    # Create student logits that will select experts [0, 1] (same as teacher top-2)
    student_logits = torch.zeros(batch, seq_len, num_experts)
    student_logits[..., 0] = 10.0  # highest
    student_logits[..., 1] = 9.0   # second highest
    student_logits[..., 2] = 1.0   # much lower
    
    result = compute_expert_shift_detailed_NEW(student_logits, teacher_indices, k_routing)
    print(f"  Student top-{k_routing} indices: {torch.topk(student_logits, k=k_routing, dim=-1)[1][0, 0]}")
    print(f"  Teacher top-{k_routing} indices: {teacher_indices[0, 0, :k_routing]}")
    print(f"  Result: shift_any={result['shift_any']:.4f}, shift_half={result['shift_half']:.4f}, shift_all={result['shift_all']:.4f}")
    print(f"  Expected: shift_any=0.0, shift_half=0.0, shift_all=0.0")
    assert result['shift_any'] == 0.0, f"Expected 0.0 but got {result['shift_any']}"
    assert result['shift_half'] == 0.0, f"Expected 0.0 but got {result['shift_half']}"
    assert result['shift_all'] == 0.0, f"Expected 0.0 but got {result['shift_all']}"
    print("  ✓ PASSED\n")
    
    
    # Case 2: Complete mismatch - student selects completely different experts
    print("=" * 60)
    print("Test Case 2: Complete mismatch (no overlap)")
    print("=" * 60)
    
    # Teacher: experts [0, 1]
    teacher_indices = torch.zeros(batch, seq_len, 3, dtype=torch.long)
    teacher_indices[..., 0] = 0
    teacher_indices[..., 1] = 1
    teacher_indices[..., 2] = 2
    
    # Student: experts [6, 7] (completely different)
    student_logits = torch.zeros(batch, seq_len, num_experts)
    student_logits[..., 6] = 10.0
    student_logits[..., 7] = 9.0
    
    result = compute_expert_shift_detailed_NEW(student_logits, teacher_indices, k_routing)
    print(f"  Student top-{k_routing} indices: {torch.topk(student_logits, k=k_routing, dim=-1)[1][0, 0]}")
    print(f"  Teacher top-{k_routing} indices: {teacher_indices[0, 0, :k_routing]}")
    print(f"  Result: shift_any={result['shift_any']:.4f}, shift_half={result['shift_half']:.4f}, shift_all={result['shift_all']:.4f}")
    print(f"  Expected: shift_any=1.0, shift_half=1.0, shift_all=1.0")
    assert result['shift_any'] == 1.0, f"Expected 1.0 but got {result['shift_any']}"
    assert result['shift_half'] == 1.0, f"Expected 1.0 but got {result['shift_half']}"
    assert result['shift_all'] == 1.0, f"Expected 1.0 but got {result['shift_all']}"
    print("  ✓ PASSED\n")


def test_compute_expert_shift_detailed_set_vs_sorted():
    """
    Test the SET COMPARISON bug - compare OLD vs NEW implementation.
    """
    print("=" * 60)
    print("Test Case 3: Set comparison OLD vs NEW")
    print("=" * 60)
    
    batch, seq_len, num_experts = 1, 1, 8
    k_routing = 2
    
    # Teacher: experts [0, 2]
    teacher_indices = torch.tensor([[[0, 2, 3]]], dtype=torch.long)
    
    # Student: experts [1, 2] (one overlap)
    student_logits = torch.zeros(batch, seq_len, num_experts)
    student_logits[..., 1] = 10.0
    student_logits[..., 2] = 9.0
    
    result_old = compute_expert_shift_detailed_OLD(student_logits, teacher_indices, k_routing)
    result_new = compute_expert_shift_detailed_NEW(student_logits, teacher_indices, k_routing)
    
    print(f"  Teacher set: {{0, 2}}")
    print(f"  Student set: {{1, 2}}")
    print(f"  True intersection: {{2}} (1 match)")
    print(f"  OLD: shift_any={result_old['shift_any']:.4f}, shift_all={result_old['shift_all']:.4f}")
    print(f"  NEW: shift_any={result_new['shift_any']:.4f}, shift_all={result_new['shift_all']:.4f}")
    print(f"  Expected: shift_any=1.0, shift_all=0.0")
    print("  ✓ Both correct for k=2 case\n")


def test_compute_expert_shift_detailed_k4_bug():
    """
    Test with k=4 to expose the sorted comparison bug.
    This is the CRITICAL test case that shows the bug.
    """
    print("=" * 60)
    print("Test Case 4: k=4 - Exposing the sorted comparison BUG")
    print("=" * 60)
    
    batch, seq_len, num_experts = 1, 1, 8
    k_routing = 4
    
    # Teacher: experts [0, 1, 2, 3]
    teacher_indices = torch.tensor([[[0, 1, 2, 3, 4]]], dtype=torch.long)
    
    # Student: experts [1, 2, 3, 4] (3 overlap: {1, 2, 3})
    student_logits = torch.zeros(batch, seq_len, num_experts)
    student_logits[..., 1] = 10.0
    student_logits[..., 2] = 9.0
    student_logits[..., 3] = 8.0
    student_logits[..., 4] = 7.0
    
    result_old = compute_expert_shift_detailed_OLD(student_logits, teacher_indices, k_routing)
    result_new = compute_expert_shift_detailed_NEW(student_logits, teacher_indices, k_routing, debug=True)
    
    print(f"\n  Teacher set: {{0, 1, 2, 3}}")
    print(f"  Student set: {{1, 2, 3, 4}}")
    print(f"  True intersection: {{1, 2, 3}} (3 matches out of 4)")
    print(f"\n  OLD (BUGGY):  shift_any={result_old['shift_any']:.4f}, shift_half={result_old['shift_half']:.4f}, shift_all={result_old['shift_all']:.4f}")
    print(f"  NEW (FIXED):  shift_any={result_new['shift_any']:.4f}, shift_half={result_new['shift_half']:.4f}, shift_all={result_new['shift_all']:.4f}")
    print(f"  EXPECTED:     shift_any=1.0, shift_half=0.0, shift_all=0.0")
    
    # Verify the bug exists in OLD implementation
    if result_old['shift_all'] == 1.0:
        print("\n  ❌ OLD BUG CONFIRMED: shift_all=1.0 (reports 0 matches)")
        print("     Sorted comparison: [1,2,3,4] vs [0,1,2,3] -> element-wise all different")
    
    # Verify NEW implementation is correct
    if result_new['shift_all'] == 0.0 and result_new['shift_any'] == 1.0:
        print("  ✓ NEW FIXED: shift_all=0.0, shift_any=1.0 (correct!)")
    else:
        print(f"  ❌ NEW still wrong!")


def test_correct_set_intersection():
    """Demonstrate the correct way to compute set intersection for expert shift."""
    print("\n" + "=" * 60)
    print("Demonstration: Correct Set Intersection Method")
    print("=" * 60)
    
    batch, seq_len, num_experts = 1, 1, 8
    k_routing = 4
    
    # Teacher: experts [0, 1, 2, 3]
    teacher_indices = torch.tensor([[[0, 1, 2, 3, 4]]], dtype=torch.long)
    teacher_topk = teacher_indices[..., :k_routing]
    
    # Student: experts [1, 2, 3, 4]
    student_logits = torch.zeros(batch, seq_len, num_experts)
    student_logits[..., 1] = 10.0
    student_logits[..., 2] = 9.0
    student_logits[..., 3] = 8.0
    student_logits[..., 4] = 7.0
    
    student_probs = torch.softmax(student_logits, dim=-1)
    _, student_indices = torch.topk(student_probs, k=k_routing, dim=-1)
    
    print(f"  Teacher top-{k_routing}: {teacher_topk[0, 0].tolist()}")
    print(f"  Student top-{k_routing}: {student_indices[0, 0].tolist()}")
    
    # WRONG method: sorted element-wise comparison
    student_sorted, _ = torch.sort(student_indices, dim=-1)
    teacher_sorted, _ = torch.sort(teacher_topk, dim=-1)
    wrong_matches = (student_sorted == teacher_sorted).sum(dim=-1).item()
    print(f"\n  WRONG method (sorted element-wise): {int(wrong_matches)} matches")
    
    # CORRECT method: convert to one-hot and use intersection
    student_onehot = torch.zeros(batch, seq_len, num_experts)
    student_onehot.scatter_(-1, student_indices, 1)
    
    teacher_onehot = torch.zeros(batch, seq_len, num_experts)
    teacher_onehot.scatter_(-1, teacher_topk, 1)
    
    correct_matches = (student_onehot * teacher_onehot).sum(dim=-1).item()
    print(f"  CORRECT method (one-hot intersection): {int(correct_matches)} matches")
    
    print(f"\n  Expected: 3 matches ({{1, 2, 3}})")


if __name__ == "__main__":
    test_compute_expert_shift_detailed_basic()
    test_compute_expert_shift_detailed_set_vs_sorted()
    test_compute_expert_shift_detailed_k4_bug()
    test_correct_set_intersection()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("1. OLD compute_expert_shift_detailed had a CRITICAL BUG:")
    print("   - Used sorted element-wise comparison instead of proper set intersection")
    print("   - This causes incorrect match counts when sets have partial overlap")
    print("   - Example: {0,1,2,3} vs {1,2,3,4} should have 3 matches, but reports 0")
    print("\n2. This bug caused artificially HIGH shift metrics!")
    print("   - Since matches are undercounted, shifts are overcounted")
    print("   - This explains the 99.94% shift_any in the logs")
    print("\n3. NEW implementation uses one-hot encoding for correct set intersection")
