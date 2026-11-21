# -*- coding: utf-8 -*-
"""
Teorik önermelerin sayısal testleri.
Çalıştır:  .venv/bin/python -m pytest code/tests -v
"""
import os, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from casper_core import (pairwise, tolerance, closure, closure_naive, gamma_op, gamma_near,
                         gamma_tol, delta_near, near_approximations, gamma_pfi_mc, gamma_pfi_boot,
                         tstat_rule, eps_grid, select_eps)

RNG = np.random.RandomState(123)


def random_tables(k=300, nmax=9, pmax=4, vmax=5):
    rng = np.random.RandomState(7)
    for _ in range(k):
        n = rng.randint(3, nmax + 1)
        p = rng.randint(1, pmax + 1)
        X = rng.randint(0, vmax, size=(n, p)).astype(float)
        y = rng.randint(0, 2, size=n)
        if len(set(y)) < 2:
            y[0] = 1 - y[1]
        yield X, y


# ---------------------------------------------------------------- karakterizasyon (Prop. char)
def test_closure_equals_definition_and_characterisation():
    """N(x)=∩{T(y): x∈T(y)} (tanım) = vektörel kapanış = {z : T(x) ⊆ T(z)}; x ∈ N(x) ⊆ T(x)."""
    for X, y in random_tables():
        D = pairwise(X)
        for eps in range(0, 5):
            A = tolerance(D, eps)
            N1, N2 = closure(A), closure_naive(A)
            assert np.array_equal(N1, N2)
            incl = np.array([[np.all(~A[x] | A[z]) for z in range(len(y))] for x in range(len(y))])
            assert np.array_equal(N1, incl)
            assert np.all(np.diag(N1))
            assert np.all(~N1 | A)  # N(x) ⊆ T(x)


# ---------------------------------------------------------------- ε limitleri
def test_limit_large_eps_gamma_op_zero_near_one():
    """ε ≥ diam: N(x)=U ⇒ γ_op = 0 (≥2 sınıf), near-γ = 1 ve δ^γ = 1."""
    for X, y in random_tables():
        D = pairwise(X)
        diam = D.max()
        assert gamma_op(D, y, diam) == 0.0
        assert gamma_near(D, y, diam) == 1.0
        assert delta_near(D, y, diam) == 1.0


def test_limit_small_eps_duplicates():
    """ε < en küçük pozitif mesafe: N(x) = x'in kopyaları; γ_op = saf kopya gruplarının oranı."""
    for X, y in random_tables():
        D = pairwise(X)
        N = closure(tolerance(D, 0.0))
        assert np.array_equal(N, D == 0)
        keys = [tuple(r) for r in X]
        pure = [len({y[j] for j in range(len(y)) if keys[j] == keys[i]}) == 1 for i in range(len(y))]
        assert gamma_op(D, y, 0.0) == pytest.approx(np.mean(pure))


# ---------------------------------------------------------------- monotonluk
def test_gamma_op_not_monotone_in_eps():
    X = np.array([[0, 1], [1, 1], [3, 0], [3, 1]], float)
    y = np.array([0, 0, 1, 0])
    g = [gamma_op(pairwise(X), y, e) for e in range(4)]
    assert g == [1.0, 0.5, 0.75, 0.75]  # düşer, sonra yükselir


def test_gamma_op_not_monotone_in_attributes():
    X1 = np.array([[1], [0], [3], [2]], float)
    X2 = np.array([[1, 0], [0, 0], [3, 0], [2, 3]], float)
    y = np.array([0, 1, 0, 0])
    assert gamma_op(pairwise(X1), y, 1) == 0.75
    assert gamma_op(pairwise(X2), y, 1) == 0.5  # öznitelik EKLENİNCE γ düştü


def test_gamma_tol_monotone_in_attributes():
    """δ-top (T(x)) bağımlılığı öznitelik eklemede azalmaz (Hu 2008 argümanı)."""
    rng = np.random.RandomState(1)
    for X, y in random_tables():
        c = rng.randint(0, 5, size=(len(y), 1)).astype(float)
        X2 = np.hstack([X, c])
        for e in range(0, 6):
            assert gamma_tol(pairwise(X2), y, e) >= gamma_tol(pairwise(X), y, e) - 1e-12


# ---------------------------------------------------------------- Thm 2.16 kapsama zinciri
def test_inclusion_chain_and_gamma_order():
    """lower ⊆ lower^γ ⊆ A ⊆ upper^γ ⊆ upper; dolayısıyla γ_op ≤ near-γ."""
    for X, y in random_tables():
        D = pairwise(X)
        for e in range(0, 5):
            N = closure(tolerance(D, e))
            for c in (0, 1):
                A = y == c
                r = near_approximations(N, A)
                assert np.all(~r["low"] | r["low_g"])
                assert np.all(~r["low_g"] | A)
                assert np.all(~A | r["up_g"])
                assert np.all(~r["up_g"] | r["up"])
            assert gamma_op(D, y, e) <= gamma_near(D, y, e) + 1e-12


# ---------------------------------------------------------------- Prop 3.1
def test_tolerance_can_be_transitive_for_intermediate_eps():
    """Makaledeki 'yalnızca iki durumda denklik' ifadesi yanlış: 0<ε<diam iken de geçişli olabilir."""
    X = np.array([[0], [0], [1], [10], [11]], float)
    D = pairwise(X)
    A = tolerance(D, 1)
    trans = np.all((A.astype(int) @ A.astype(int) > 0) <= A)
    assert 0 < 1 < D.max() and trans


# ---------------------------------------------------------------- Gamma-PFI özellikleri
@pytest.mark.parametrize("g", ["op", "near"])
def test_gamma_pfi_bounded_dummy_and_shift_invariant(g):
    rng = np.random.RandomState(3)
    for X, y in list(random_tables(60, nmax=10)):
        Xc = np.hstack([X, np.full((len(y), 1), 2.0)])  # sabit sütun
        eps = float(np.median(pairwise(Xc)))
        mu, sd, sem, deltas, _ = gamma_pfi_mc(Xc, y, eps, g, n_rep=8, rng=np.random.RandomState(0))
        assert np.all(np.abs(deltas) <= 1.0)
        assert np.all(deltas[-1] == 0) and sd[-1] == 0          # dummy: Δγ ≡ 0
        Xs = Xc.copy(); Xs[:, 0] += 5                            # sıra-koruyan öteleme
        mu2, *_ = gamma_pfi_mc(Xs, y, eps, g, n_rep=8, rng=np.random.RandomState(0))
        assert np.allclose(mu, mu2)


def test_single_permutation_changes_distance_by_at_most_range():
    """Tek sütun permütasyonu her çift mesafesini en fazla max_f − min_f (= k_f − 1) değiştirir."""
    for X, y in random_tables():
        D = pairwise(X)
        for i in range(X.shape[1]):
            Xp = X.copy(); Xp[:, i] = RNG.permutation(X[:, i])
            assert np.abs(pairwise(Xp) - D).max() <= X[:, i].max() - X[:, i].min() + 1e-12


# ---------------------------------------------------------------- t_f asimptotiği
def test_mc_sem_shrinks_and_rule_converges_to_quartile_rule():
    """Sabit E[Δγ_f] ve sabit varyans altında SEM ~ 1/√N; N→∞ iken t-kuralı {μ_f > Q1} kuralına yakınsar,
    yani κ etkisizleşir."""
    rng = np.random.RandomState(0)
    true_mu = np.array([0.05, 0.02, 0.01, 0.0, -0.01, 0.03, 0.006, 0.015])  # Q1'den ayrık (eşitlik yok)
    q1 = np.percentile(true_mu, 25)
    target = set(np.where(true_mu > q1)[0])
    sems = []
    for N in (10, 100, 100000):
        d = true_mu[:, None] + 0.05 * rng.randn(len(true_mu), N)
        mu, sem = d.mean(1), d.std(1) / np.sqrt(N)
        sems.append(sem.mean())
        sel_k1 = set(tstat_rule(mu, sem, 1.0)[0]); sel_k3 = set(tstat_rule(mu, sem, 3.0)[0])
    assert sems[0] > sems[1] > sems[2]
    assert sems[0] / sems[2] == pytest.approx(np.sqrt(10000), rel=0.25)
    assert sel_k1 == target and sel_k3 == target   # N büyükken κ=1 ve κ=3 aynı kümeyi verir


def test_bootstrap_se_does_not_vanish_with_B():
    rng = np.random.RandomState(5)
    n = 60
    X = rng.randint(0, 4, size=(n, 3)).astype(float)
    y = (X[:, 0] + rng.randint(0, 2, n) > 2).astype(int)
    eps = 2.0
    se20, _ = gamma_pfi_boot(X, y, eps, "op", n_boot=20, rng=np.random.RandomState(1))
    se80, _ = gamma_pfi_boot(X, y, eps, "op", n_boot=80, rng=np.random.RandomState(1))
    assert se80[0] > 0.5 * se20[0]   # MC-SEM gibi 1/√4 = 0.5 oranında küçülmez


# ---------------------------------------------------------------- ε kuralı
def test_eps_rule_excludes_trivial_plateau():
    rng = np.random.RandomState(2)
    X = rng.randint(0, 5, size=(40, 4)).astype(float)
    y = (X[:, 0] > 2).astype(int)
    D = pairwise(X)
    e, g, grid, curve = select_eps(D, y, gamma_op)
    assert g < 0.999 and e in grid


# ----------------------------------------------------------------- Önerme prop:mem (CASPER-μF)
def test_mem_properties():
    """(i) γ^μ∈[0,1], γ^μ ≥ γ^T, eşitlik ⇔ γ^T=1; (ii) ε≥diam ⇒ Σπ_c²; (iii) ε<min pozitif mesafe ⇒ 1."""
    from casper_core import gamma_mem, gamma_tol
    rng = np.random.RandomState(3)
    for _ in range(200):
        n = rng.randint(4, 15); X = rng.randint(1, 5, (n, 3)).astype(float); y = rng.randint(0, 2, n)
        D = pairwise(X)
        for e in range(0, int(D.max()) + 2):
            gm, gt = gamma_mem(D, y, e), gamma_tol(D, y, e)
            assert -1e-12 <= gm <= 1 + 1e-12 and gm >= gt - 1e-12
            assert (abs(gm - 1) < 1e-12) == (abs(gt - 1) < 1e-12)
        pi = np.bincount(y, minlength=2) / n
        assert abs(gamma_mem(D, y, D.max()) - (pi ** 2).sum()) < 1e-12
        Xu, idx = np.unique(X, axis=0, return_index=True)   # kopyasız: ε=0 ⇒ 1
        if len(idx) > 2:
            assert abs(gamma_mem(pairwise(X[idx]), y[idx], 0) - 1) < 1e-12


def test_mem_not_monotone():
    """(iv) karşı örnek: öznitelik eklemek γ^μ'yü düşürebilir (a=(1,1,2), b=(0,1,0), y=(1,0,0), ε=1)."""
    from casper_core import gamma_mem
    X = np.array([[1., 0.], [1., 1.], [2., 0.]]); y = np.array([1, 0, 0])
    a, b = gamma_mem(pairwise(X[:, :1]), y, 1), gamma_mem(pairwise(X), y, 1)
    assert abs(a - 5 / 9) < 1e-12 and abs(b - 4 / 9) < 1e-12
