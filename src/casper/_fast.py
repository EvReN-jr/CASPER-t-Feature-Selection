"""Exact fast evaluation of the CASPER-t dependency degree and gamma-PFI.

The functions here return bit-for-bit the same epsilon, gamma, Delta_gamma
and SEM as the reference implementation in ``_core`` (same random stream,
same floating point distances, same integer counts).  They only avoid work
that cannot change the result.

Three facts make this possible.

1. Objects with the same code vector have the same row of the tolerance
   matrix A, hence the same T, the same minimal neighbourhood N and the same
   lower-approximation status.  Gamma can be computed on the distinct
   patterns and weighted by their multiplicities.  A pattern holding objects
   of two classes is never in the lower approximation (each of its objects
   lies in the other's N).  Under a column permutation the patterns are
   rebuilt from (pattern of the other columns, permuted code), and the
   distances are D_rest + |a - b|.

2. A is reflexive and symmetric, so T(x) <= T(z) implies z in T(x).  An
   object x is therefore in the lower approximation iff no z in T(x) of
   another class satisfies T(x) <= T(z).  Only those candidate pairs are
   tested, with packed bit rows and early exit (Numba), or with a matrix
   product restricted to the other-class columns (NumPy fallback).

3. Gamma is a count divided by n, so it does not depend on the order of the
   objects; the dense path sorts them by class once.

Without Numba the NumPy fallback is used; it is also exact, only slower.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

try:  # optional accelerator
    from numba import njit, prange
    HAVE_NUMBA = True
except Exception:  # pragma: no cover - exercised when numba is absent
    HAVE_NUMBA = False

_EMPTY = np.zeros(0)
DENSE_RATIO = 0.5   # distinct patterns / objects above which patterns are not merged


if HAVE_NUMBA:
    @njit(cache=True, parallel=True)
    def _nb_lower(d, perm, cls, eps):
        """lower[x] for A[x,z] = (d[x,z] + |perm[x]-perm[z]|) <= eps, diagonal
        True.  ``perm`` of length 0 means no permuted column.  cls[x] < 0 marks
        a pattern that holds several classes (never lower)."""
        u = d.shape[0]
        nw = (u + 63) // 64
        W = np.zeros((u, nw), np.uint64)
        use_p = perm.shape[0] > 0
        for x in prange(u):
            for z in range(u):
                v = d[x, z]
                if use_p:
                    v = v + abs(perm[x] - perm[z])
                if v <= eps or x == z:
                    W[x, z >> 6] |= np.uint64(1) << np.uint64(z & 63)
        low = np.zeros(u, np.bool_)
        for x in prange(u):
            cx = cls[x]
            if cx < 0:
                continue
            ok = True
            for z in range(u):
                if cls[z] == cx:
                    continue
                if (W[x, z >> 6] >> np.uint64(z & 63)) & np.uint64(1):
                    sub = True
                    for w in range(nw):
                        if W[x, w] & ~W[z, w]:
                            sub = False
                            break
                    if sub:
                        ok = False
                        break
            low[x] = ok
        return low


def _np_lower(d, cls, eps):
    """NumPy fallback of ``_nb_lower`` (no permuted column)."""
    A = d <= eps
    np.fill_diagonal(A, True)
    lower = cls >= 0
    Af = A.astype(np.float32)
    nAf = (~A).astype(np.float32)
    for c in np.unique(cls[cls >= 0]):
        rows = np.flatnonzero(cls == c)
        cols = np.flatnonzero(cls != c)
        if len(cols) == 0:
            continue
        sub = A[np.ix_(rows, cols)]
        has = sub.any(axis=1)          # some other-class pattern in T(P)
        r2 = rows[has]
        if len(r2) == 0:
            continue
        c2 = cols[sub[has].any(axis=0)]
        nr = (Af[r2] @ nAf[:, c2]) < 0.5   # T(P) <= T(Q)
        lower[r2[nr.any(axis=1)]] = False
    return lower


def _lower(d, cls, eps, perm=None):
    if HAVE_NUMBA:
        p = _EMPTY if perm is None else np.ascontiguousarray(perm, dtype=np.float64)
        return _nb_lower(np.ascontiguousarray(d, dtype=np.float64), p,
                         np.ascontiguousarray(cls, dtype=np.int64), float(eps))
    if perm is not None:
        d = d + np.abs(perm[:, None] - perm[None, :])
    return _np_lower(d, cls, eps)


def _pattern_classes(pid, yi, u, k):
    cnt = np.bincount(pid * k + yi, minlength=u * k).reshape(u, k)
    cls = np.where((cnt > 0).sum(1) == 1, cnt.argmax(1), -1)
    return cls, cnt.sum(1)


def _labels(y):
    _, yi = np.unique(y, return_inverse=True)
    yi = yi.ravel().astype(np.int64)
    return yi, int(yi.max()) + 1


def _unique_rows(X):
    U, pid = np.unique(X, axis=0, return_inverse=True)
    return U, pid.ravel()


def eps_range(X, n_points=25):
    """Same candidates as ``_core.eps_range``."""
    d = cdist(X, X, "cityblock")
    upper = d[np.triu_indices(len(X), 1)]
    cand = np.unique(np.round(np.percentile(upper, np.linspace(5, 95, n_points))).astype(int))
    return sorted(int(e) for e in cand if e > 0)


def select_epsilon(X, y):
    """Exact fast version of ``_core.select_epsilon``."""
    X = np.asarray(X, dtype=float)
    n = len(y)
    yi, k = _labels(y)
    U, pid = _unique_rows(X)
    if len(U) > DENSE_RATIO * n:
        d, cls, mult = cdist(X, X, "cityblock"), yi, np.ones(n, dtype=np.int64)
    else:
        cls, mult = _pattern_classes(pid, yi, len(U), k)
        d = cdist(U, U, "cityblock")
    best_eps, best_gamma = None, -1.0
    fallback_eps, fallback_gamma = None, -1.0
    for eps in eps_range(X):
        g = float(mult[_lower(d, cls, eps)].sum()) / n
        if g > fallback_gamma:
            fallback_eps, fallback_gamma = eps, g
        if g < 0.999 and g > best_gamma:
            best_eps, best_gamma = eps, g
    if best_eps is None:
        best_eps, best_gamma = fallback_eps, fallback_gamma
    return int(best_eps), float(best_gamma)


def _small_int(X):
    return bool(np.all(X == np.rint(X)) and (X.size == 0 or np.abs(X).max() < 2 ** 20))


def gamma_pfi(D, y, eps_star, gamma_base, n_rep=30, random_state=42, rng=None,
              return_deltas=False):
    """Exact fast version of ``_core.gamma_pfi``: same permutations in the same
    order, same Delta_gamma replicates, same mean and SEM.  ``rng`` (a
    RandomState) overrides ``random_state``; ``return_deltas`` also returns the
    (p, n_rep) matrix of replicates."""
    D = np.asarray(D, dtype=float)
    n, p = D.shape
    rng = np.random.RandomState(random_state) if rng is None else rng
    deltas = np.empty((p, n_rep))
    yi, k = _labels(y)
    U, gfull = _unique_rows(D)
    exact_int = _small_int(D)
    mean_dg = np.empty(p)
    sem_dg = np.empty(p)

    if len(U) > DENSE_RATIO * n:
        # object level; d_perm = (d_base - col_i) + |perm_i - perm_j| as in _core
        o = np.argsort(yi, kind="stable")
        ys = yi[o]
        Ds = D[o]
        d_base = cdist(Ds, Ds, "cityblock")
        for i in range(p):
            base_i = d_base - np.abs(Ds[:, i][:, None] - Ds[:, i][None, :])
            drops = np.empty(n_rep)
            for r in range(n_rep):
                perm = rng.permutation(D[:, i])[o]
                cnt = int(_lower(base_i, ys, eps_star, perm).sum())
                drops[r] = gamma_base - float(cnt) / n
            deltas[i] = drops
            mean_dg[i] = drops.mean()
            sem_dg[i] = drops.std() / np.sqrt(n_rep)
        return (mean_dg, sem_dg, deltas) if return_deltas else (mean_dg, sem_dg)

    if not exact_int:
        d_base = cdist(D, D, "cityblock")
    for i in range(p):
        if exact_int and p > 1:
            # integer codes: d_base - col_i equals the L1 distance on the other
            # columns exactly, so objects are grouped by those columns only
            Um, g = _unique_rows(np.delete(D, i, axis=1))
            Dm = cdist(Um, Um, "cityblock")
        elif exact_int:
            g, Dm = np.zeros(n, dtype=np.int64), np.zeros((1, 1))
        else:
            # general floats: group by the full row so that every member of a
            # group shares the same d_base - col_i entries bit for bit
            g = gfull
            col_i = np.abs(D[:, i][:, None] - D[:, i][None, :])
        vals = np.unique(D[:, i])
        nv = len(vals)
        drops = np.empty(n_rep)
        for r in range(n_rep):
            perm = rng.permutation(D[:, i])
            key = g * nv + np.searchsorted(vals, perm)
            uk, pid = np.unique(key, return_inverse=True)
            pid = pid.ravel()
            cls, mult = _pattern_classes(pid, yi, len(uk), k)
            av = vals[uk % nv]
            if exact_int:
                base = Dm[np.ix_(uk // nv, uk // nv)]
            else:
                rep = np.empty(len(uk), dtype=np.int64)
                rep[pid] = np.arange(n)          # any member represents its group
                base = d_base[np.ix_(rep, rep)] - col_i[np.ix_(rep, rep)]
            low = _lower(base, cls, eps_star, av)
            drops[r] = gamma_base - float(mult[low].sum()) / n
        deltas[i] = drops
        mean_dg[i] = drops.mean()
        sem_dg[i] = drops.std() / np.sqrt(n_rep)
    return (mean_dg, sem_dg, deltas) if return_deltas else (mean_dg, sem_dg)
