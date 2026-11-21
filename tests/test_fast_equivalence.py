"""The fast engine (_fast) must reproduce the reference engine (_core) bit for
bit: same epsilon*, same baseline gamma, same Delta_gamma and SEM arrays."""
import numpy as np
import pytest

from casper import _core, _fast


def _tables(seed, count):
    rng = np.random.default_rng(seed)
    for t in range(count):
        n = int(rng.integers(5, 150))
        p = int(rng.integers(1, 7))
        X = rng.integers(1, int(rng.integers(2, 8)) + 1, size=(n, p)).astype(float)
        if t % 3 == 0:                       # non-integer pass-through column
            X[:, 0] = rng.choice([0.0, 2.7], n)
        if t % 4 == 0:                       # many duplicate rows
            X = X[rng.integers(0, max(2, n // 6), n)]
        y = rng.integers(0, int(rng.integers(2, 4)), n).astype(float)
        yield t, X, y


@pytest.mark.parametrize("dense_ratio", [0.5, 0.0, 2.0])   # default / always dense / always patterns
def test_gamma_pfi_bit_identical(monkeypatch, dense_ratio):
    monkeypatch.setattr(_fast, "DENSE_RATIO", dense_ratio)
    for t, X, y in _tables(int(dense_ratio * 10) + 1, 120):
        if not _core.eps_range(X):
            continue
        e_ref = _core.select_epsilon(X, y)
        assert _fast.select_epsilon(X, y) == e_ref
        a = _core.gamma_pfi(X, y, *e_ref, n_rep=6, random_state=t)
        b = _fast.gamma_pfi(X, y, *e_ref, n_rep=6, random_state=t)
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_numpy_fallback_bit_identical(monkeypatch):
    monkeypatch.setattr(_fast, "HAVE_NUMBA", False)
    for t, X, y in _tables(7, 80):
        if not _core.eps_range(X):
            continue
        e_ref = _core.select_epsilon(X, y)
        assert _fast.select_epsilon(X, y) == e_ref
        a = _core.gamma_pfi(X, y, *e_ref, n_rep=5, random_state=t)
        b = _fast.gamma_pfi(X, y, *e_ref, n_rep=5, random_state=t)
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_gamma_on_patterns_matches_definition():
    rng = np.random.default_rng(3)
    for _ in range(200):
        n = int(rng.integers(3, 120))
        X = rng.integers(1, 5, size=(n, int(rng.integers(1, 5)))).astype(float)
        y = rng.integers(0, 3, n)
        d = _core.l1(X)
        for eps in (0, 1, 2, 3.5):
            ref = _core.gamma_from_dmat(d, y, eps)
            assert float(_fast._lower(d, y, eps).sum()) / n == ref
            U, pid = _fast._unique_rows(X)
            cls, mult = _fast._pattern_classes(pid, y.astype(np.int64), len(U), 3)
            assert float(mult[_fast._lower(_core.l1(U), cls, eps)].sum()) / n == ref
