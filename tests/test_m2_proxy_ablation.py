"""Deterministic ablation: equal present value yet F can misrank future acceptance.

Run: python -m pytest -q test_m2_proxy_ablation.py
The future scheduler is the same sequential single-resource Best-Fit in both states.
"""
from itertools import product
import pytest

THRESHOLDS = (1, 2, 4, 8)
WEIGHTS = (1, 1, 2, 4)

def score(residual):
    return sum(w * sum(r >= q for r in residual)
               for q, w in zip(THRESHOLDS, WEIGHTS))

def admitted(residual, arrivals):
    remaining = list(residual)
    accepted = 0
    for demand in arrivals:
        feasible = [i for i, free in enumerate(remaining) if free >= demand]
        if feasible:
            i = min(feasible, key=lambda k: (remaining[k] - demand, k))
            remaining[i] -= demand
            accepted += 1
    return accepted

def test_fixed_current_value_and_capacity_proxy_reversal():
    # From the SAME two accepted current jobs: 7 GPU and 1 GPU on 8+8 nodes.
    # A packs them together; B splits them. Both achieve the same present value.
    A, B = (0, 8), (1, 7)
    assert sum(A) == sum(B) == 8
    assert score(A) == 8 > score(B) == 5
    future = (8, 2, 2)
    assert admitted(A, future) == 1 < admitted(B, future) == 2

@pytest.mark.parametrize("demands,expected", [
    ((1, 2, 4, 8), (64, 129, 130, 7, 8, 49)),
    ((2, 4, 8), (27, 47, 43, 7, 3, 17)),
    ((1, 2), (8, 24, 24, 0, 0, 8)),
    ((4, 8), (8, 11, 7, 4, 0, 4)),
])
def test_future_distribution_sensitivity(demands, expected):
    # Exhaust ALL ordered 3-job future batches over the specified support.
    states = [(admitted((0, 8), seq), admitted((1, 7), seq))
              for seq in product(demands, repeat=3)]
    observed = (len(states),
                sum(a for a, b in states), sum(b for a, b in states),
                sum(a > b for a, b in states), sum(b > a for a, b in states),
                sum(a == b for a, b in states))
    assert observed == expected
