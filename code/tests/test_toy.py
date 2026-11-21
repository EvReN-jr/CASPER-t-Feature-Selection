# -*- coding: utf-8 -*-
"""Oyuncak örneğin sayılarını TANIMDAN, kütüphane fonksiyonlarına güvenmeden yeniden hesaplar."""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import toy_example as T

R = T.run()
X, y = T.CODES, T.Y
n = len(y)


def l1(i, j):
    return sum(abs(X[i, k] - X[j, k]) for k in range(X.shape[1]))


def test_tolerance_classes_by_hand():
    e = R["eps_star"]
    for i in range(n):
        assert R["T"][i] == [j for j in range(n) if l1(i, j) <= e]


def test_minimal_neighbourhood_by_hand():
    Tc = [set(t) for t in R["T"]]
    for x in range(n):
        Nx = set(range(n))
        for yy in range(n):
            if x in Tc[yy]:
                Nx &= Tc[yy]
        assert sorted(Nx) == R["N"][x]


def test_gamma_base_by_hand():
    pos = [all(y[z] == y[x] for z in R["N"][x]) for x in range(n)]
    assert pos == R["pos"] and sum(pos) / n == R["gamma_base"] == 0.875


def test_mc_mean_consistent_with_exact_moments():
    for f in range(3):
        assert abs(R["mu"][f] - R["exact_mean"][f]) <= 3 * R["exact_sd"][f] / np.sqrt(30)


def test_t_and_selection_by_hand():
    mu = np.array(R["mu"]); sem = np.array(R["sem"])
    q1 = np.percentile(mu, 25)
    t = (mu - q1) / (sem + 1e-9)
    assert np.allclose(t, R["t"]) and R["selected"] == ["a"]


def test_near_gamma_saturates_on_toy():
    """near-γ bu örnekte her ε'da 1: near-γ ile ε seçimi dejenere olur."""
    assert all(abs(h - 1.0) < 1e-12 for _, _, h in R["scan"])
