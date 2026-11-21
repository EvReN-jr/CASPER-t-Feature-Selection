"""Core numerics for CASPER-t: L1 tolerance neighbourhoods, the gamma
dependency degree, epsilon selection, and the reliability-weighted (t_f)
elimination rule.

These are extracted verbatim (in behaviour) from the paper's reference
pipeline; see the accompanying article, Sections 2--3.  They serve as the
reference for the exact fast engine in ``_fast``, which ``CasperT`` uses.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

RANDOM_STATE = 42
N_REP = 30          # gamma-PFI permutation replicates per feature
N_EPS_PTS = 25      # points in the epsilon sweep
MAX_N_TOPOLOGICAL = 3000   # subsample cap for the O(n^2) topological stage
KAPPA = 1.5         # reliability margin of the t_f rule (Eq. 22)


def l1(X):
    """City-block (Manhattan / L1) distance matrix over the (discretized) rows."""
    return cdist(np.asarray(X, dtype=float), np.asarray(X, dtype=float), "cityblock")


def _neighborhood_closure(adj_with_diag):
    """nr[i, j] = True iff every tolerance-neighbour k of i is also a neighbour
    of j.  BLAS matrix product replaces the Python double loop."""
    A = adj_with_diag.astype(np.float32)
    not_nr = (A @ (~adj_with_diag).astype(np.float32)) > 0.5
    return ~not_nr


def gamma_from_dmat(d_mat, y, eps):
    """Gamma dependency degree (lower-approximation accuracy) at threshold eps:
    the fraction of objects whose entire tolerance neighbourhood is class-pure."""
    n = len(y)
    adj = d_mat <= eps
    np.fill_diagonal(adj, True)
    same_class = (y[:, None] == y[None, :])
    nr = _neighborhood_closure(adj)
    lower_mask = np.all((~nr) | same_class, axis=1)
    return float(lower_mask.sum()) / n


def eps_range(X, n_points=N_EPS_PTS):
    """Data-driven epsilon candidates from the 5th--95th percentiles of the
    pairwise L1 distances."""
    d = l1(X)
    upper = d[np.triu_indices(len(X), 1)]
    cand = np.unique(np.round(np.percentile(upper, np.linspace(5, 95, n_points))).astype(int))
    return sorted(int(e) for e in cand if e > 0)


def select_epsilon(X, y):
    """Deterministic optimal-epsilon rule: the eps maximizing gamma subject to
    gamma < 0.999 (the collapse guard).  Returns (eps_star, gamma_base)."""
    best_eps, best_gamma = None, -1.0
    fallback_eps, fallback_gamma = None, -1.0
    for eps in eps_range(X):
        g = gamma_from_dmat(l1(X), y, eps)
        if g > fallback_gamma:
            fallback_eps, fallback_gamma = eps, g
        if g < 0.999 and g > best_gamma:
            best_eps, best_gamma = eps, g
    if best_eps is None:            # every candidate saturated -> take the max
        best_eps, best_gamma = fallback_eps, fallback_gamma
    return int(best_eps), float(best_gamma)


def stratified_subsample(y, max_n, random_state=RANDOM_STATE):
    """Class-proportional subsample index; only the O(n^2) topological stage
    uses it, never the downstream evaluation."""
    n = len(y)
    if n <= max_n:
        return np.arange(n)
    rng = np.random.RandomState(random_state)
    idx = []
    for c in np.unique(y):
        pool = np.flatnonzero(y == c)
        take = max(1, int(round(len(pool) * max_n / n)))
        idx.append(rng.choice(pool, size=min(take, len(pool)), replace=False))
    return np.sort(np.concatenate(idx))


def gamma_pfi(D, y, eps_star, gamma_base, n_rep=N_REP, random_state=RANDOM_STATE):
    """Gamma permutation feature importance.  For each column, permute it
    ``n_rep`` times, recompute gamma, and record the drop Delta_gamma and its
    standard error.  Returns arrays (mean_dg, sem_dg) per column."""
    D = np.asarray(D, dtype=float)
    n, p = D.shape
    rng = np.random.RandomState(random_state)
    d_base = l1(D)
    mean_dg = np.empty(p)
    sem_dg = np.empty(p)
    for i in range(p):
        col_i = np.abs(D[:, i][:, None] - D[:, i][None, :])
        drops = np.empty(n_rep)
        for r in range(n_rep):
            perm = rng.permutation(D[:, i])
            d_perm = d_base - col_i + np.abs(perm[:, None] - perm[None, :])
            drops[r] = gamma_base - gamma_from_dmat(d_perm, y, eps_star)
        mean_dg[i] = drops.mean()
        sem_dg[i] = drops.std() / np.sqrt(n_rep)
    return mean_dg, sem_dg


def tstat_support(mean_dg, sem_dg, kappa=KAPPA):
    """The CASPER-t reliability-weighted elimination rule (Eq. 22):
    retain feature f iff  t_f = (Delta_gamma_f - Q1) / SEM_f > kappa,
    where Q1 is the first quartile of the Delta_gamma distribution across
    features.  Guarantees at least one feature (the top-ranked by Delta_gamma).

    When SEM_f is (near) zero, for example for a constant column whose every
    permutation is a no-op, there is no usable uncertainty estimate.  The rule
    then uses the point estimate alone: the feature is retained only if its
    mean drop is positive, and never merely because Q1 is negative.
    """
    mean_dg = np.asarray(mean_dg, float)
    sem_dg = np.asarray(sem_dg, float)
    q1 = float(np.percentile(mean_dg, 25))

    zero_sem = sem_dg <= 1e-9
    safe_sem = np.where(zero_sem, 1.0, sem_dg)  # placeholder denominator, unused where zero_sem
    t = np.where(zero_sem,
                 np.where(mean_dg > 0, np.inf, -np.inf),
                 (mean_dg - q1) / safe_sem)

    mask = t > kappa
    if not mask.any():
        mask = np.zeros_like(mask)
        mask[int(np.argmax(mean_dg))] = True
    return mask, q1, t
