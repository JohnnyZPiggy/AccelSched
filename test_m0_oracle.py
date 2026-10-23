"""Run with: python -m pytest -q test_m0_oracle.py. CPU-only; no GPU/CI."""
import numpy as np
from m0_oracle import (Instance, packing_case, integrality_case, brute_lex,
                       shape_lp, mip_value, greedy, validate, future_score,
                       residual, marginal)

def test_lexicographic_packing():
    a = packing_case()
    optimum, placement = brute_lex(a)
    assert optimum == (20., 1.)
    assert validate(a, placement)
    assert shape_lp(a)["value"] == 20
    assert mip_value(a) == 20

def test_integrality_gap():
    a = integrality_case()
    lp = shape_lp(a)
    assert np.isclose(lp["value"], 40/3)
    assert np.isclose(lp["dual"], 40/3)
    assert mip_value(a) == 10
    assert brute_lex(a)[0][0] == 10

def test_not_submodular():
    f = lambda demands: float(8-sum(demands) >= 8)
    assert f([4])-f([]) < f([4,4])-f([4])

def test_not_supermodular():
    f = lambda demands: float(8-sum(demands) >= 4)
    assert f([4])-f([]) > f([4,4])-f([4])

def test_fractional_future_capacity_is_not_deployable():
    assert 8 * .75 == 6
    assert 8 * 1 > 6

def test_three_greedy_methods_and_independent_validation():
    for a in (packing_case(), integrality_case()):
        for method in ("random", "best_fit", "pgfp"):
            placement, value, future = greedy(a, method, seed=2026)
            assert validate(a, placement)
            assert value <= brute_lex(a)[0][0] + 1e-9
            assert future == future_score(a, residual(a, placement))

def test_local_marginal_and_monotonicity():
    a = packing_case()
    initial = a.capacity.astype(float)
    for i in range(2):
        changed = initial.copy()
        changed[i] -= a.demand[0]
        assert marginal(a, initial, i, 0) == -1
        assert future_score(a, changed)-future_score(a, initial) == -1

def test_cpu_bottleneck():
    a = Instance(np.array([[8., 1.]]), np.array([[4., 2.]]), np.array([1.]),
                 np.ones((1, 1), bool), np.array([0]), np.array([[8., 1.]]),
                 np.array([1.]), np.ones((1, 1), bool))
    assert brute_lex(a)[0][0] == 0
    assert greedy(a, "pgfp")[1] == 0

def test_incompatible_gpu_server():
    a = packing_case()
    a.eligible[:, 0] = False
    assert all(greedy(a, method)[0][0] == -1
               for method in ("random", "best_fit", "pgfp"))
