# -*- coding: utf-8 -*-
"""İstatistik yardımcıları."""
import numpy as np
from scipy import stats


def nb_corrected_t(diff, test_train_ratio=0.25):
    """Nadeau & Bengio (2003) düzeltilmiş tekrarlı-örnekleme t-testi. diff: J katın farkları (5×2 → J=10,
    n_test/n_train = 1/4). Döner (t, p iki yanlı)."""
    d = np.asarray(diff, float)
    J = len(d)
    s2 = d.var(ddof=1)
    if s2 == 0:
        return (0.0, 1.0) if d.mean() == 0 else (np.inf * np.sign(d.mean()), 0.0)
    t = d.mean() / np.sqrt((1.0 / J + test_train_ratio) * s2)
    return float(t), float(2 * stats.t.sf(abs(t), J - 1))


def wilcoxon_p(diff):
    d = np.asarray(diff, float)
    if np.allclose(d, 0):
        return 1.0
    try:
        return float(stats.wilcoxon(d, zero_method="zsplit").pvalue)
    except ValueError:
        return 1.0


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)


def friedman_iman_davenport(M):
    """M: (datasets × methods) skor matrisi (büyük = iyi). Döner dict: ortalama sıralar, χ², p, F_ID, p_ID, CD(α=.05)."""
    M = np.asarray(M, float)
    N, k = M.shape
    R = np.apply_along_axis(lambda r: stats.rankdata(-r), 1, M)
    Rm = R.mean(0)
    chi2 = 12 * N / (k * (k + 1)) * (np.sum(Rm ** 2) - k * (k + 1) ** 2 / 4)
    p = stats.chi2.sf(chi2, k - 1)
    F = (N - 1) * chi2 / (N * (k - 1) - chi2) if N * (k - 1) - chi2 > 0 else np.inf
    pF = stats.f.sf(F, k - 1, (k - 1) * (N - 1))
    q = stats.studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2)
    cd = q * np.sqrt(k * (k + 1) / (6 * N))
    return dict(mean_ranks=Rm, chi2=chi2, p=p, F_ID=F, p_ID=pF, CD=cd, N=N, k=k)


def bayes_signed_rank(a, b, rope=0.01):
    """Benavoli et al. (2017) Bayesçi işaretli-sıra testi (baycomp). Döner (P(a>b), P(rope), P(b>a))."""
    try:
        import baycomp
        return tuple(float(v) for v in baycomp.two_on_multiple(np.asarray(a), np.asarray(b), rope=rope))
    except Exception:
        return (np.nan, np.nan, np.nan)
