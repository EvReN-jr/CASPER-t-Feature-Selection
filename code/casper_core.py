# -*- coding: utf-8 -*-
"""
CASPER-t çekirdeği.

Orijinal pipeline (../pipeline/gamma_pfi_uhem.py)
DEĞİŞTİRİLMEZ. Bu modül:
  * orijinal γ hesabını (γ_op) birebir yeniden uygular ve testlerle orijinale eşitliğini doğrular,
  * makalenin Def. 2.14/2.15'teki near-γ (pre ∪ semi alt yaklaşım) pozitif bölgesini ekler,
  * δ-top (T(x), Hu 2008 NRS) γ'sını ablasyon için ekler,
  * Monte-Carlo SEM (orijinal) ve satır-bootstrap SE (yeni) skorlamasını sağlar,
  * ablasyon seçenekleri: metrik (L1/L2/Linf), ε kuralı, eleme kuralı, silme-önemi.

Notasyon: D (n×n) çift mesafe matrisi, A = (D <= eps) tolerans ilişkisi (yansımalı, simetrik),
T(x) = {y : A[x,y]}, minimal kaplama komşuluğu N(x) = ∩{T(y) : x ∈ T(y)} = {z : T(x) ⊆ T(z)}.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

import casper_fast

# ---------------------------------------------------------------------------
# 1. Mesafeler ve tolerans yapıları
# ---------------------------------------------------------------------------

def pairwise(X, metric="l1", nominal=None):
    """Çift mesafe matrisi. metric ∈ {'l1','l2','linf'}.
    nominal (bool maske, yalnız l1): HEOM-tipi karışık mesafe — sayısal (Jenks kodu) sütunlarda |a−b|,
    nominal sütunlarda 1[a≠b] (bir bin adımı ağırlığında Hamming). Karışık tip uzantısı."""
    if nominal is not None and np.any(nominal):
        assert metric == "l1"
        return sum(_col_dist(X[:, i], metric, nominal[i]) for i in range(X.shape[1]))
    name = {"l1": "cityblock", "l2": "euclidean", "linf": "chebyshev"}[metric]
    return cdist(X, X, name)


def tolerance(D, eps):
    A = D <= eps + 1e-12
    np.fill_diagonal(A, True)
    return A


def closure(A):
    """
    nr[i,j] = True  ⇔  j ∈ N(i)  ⇔  T(i) ⊆ T(j).
    Orijinal `_neighborhood_closure` ile aynı: ¬nr = A @ ¬A > 0 (boolean matris çarpımı).
    Maliyet O(n^3) (BLAS ile pratikte O(n^ω)).
    """
    Af = A.astype(np.float32)
    not_nr = (Af @ (~A).astype(np.float32)) > 0.5
    return ~not_nr


def closure_naive(A):
    """Tanımdan döngüyle N(x) = ∩_{y : x∈T(y)} T(y). Yalnızca testler için."""
    n = A.shape[0]
    N = np.zeros_like(A)
    for x in range(n):
        acc = np.ones(n, dtype=bool)
        for y in range(n):
            if A[y, x]:
                acc &= A[y]
        N[x] = acc
    return N


# ---------------------------------------------------------------------------
# 2. Bağımlılık dereceleri
# ---------------------------------------------------------------------------

def gamma_op(D, y, eps):
    """Operasyonel γ (orijinal kod): |{x : N(x) ⊆ [x]_D}| / |U|."""
    N = closure(tolerance(D, eps))
    same = y[:, None] == y[None, :]
    return float(np.all(~N | same, axis=1).mean())


def gamma_tol(D, y, eps):
    """δ-top / tolerans sınıfı γ (Hu et al. 2008 NRS): |{x : T(x) ⊆ [x]_D}| / |U|."""
    A = tolerance(D, eps)
    same = y[:, None] == y[None, :]
    return float(np.all(~A | same, axis=1).mean())


def near_approximations(N, inA):
    """
    Makale Def. 2.14 — tek komşuluk N(x) için:
      lower(A) = {x : N(x) ⊆ A},  upper(A) = {x : N(x) ∩ A ≠ ∅}
      lower^p  = {x ∈ A : N(x) ⊆ upper(A)}
      upper^p  = A ∪ {x ∉ A : N(x) ∩ lower(A) ≠ ∅}
      lower^s  = {x ∈ A : N(x) ∩ lower(A) ≠ ∅}
      upper^s  = A ∪ {x ∈ upper(A) : N(x) ⊆ upper(A)}
      lower^γ = lower^p ∪ lower^s,  upper^γ = upper^p ∩ upper^s
    N: (n×n) bool, N[x,z] ⇔ z ∈ N(x). inA: (n,) bool.
    """
    low = np.all(~N | inA[None, :], axis=1)
    up = np.any(N & inA[None, :], axis=1)
    hits_low = np.any(N & low[None, :], axis=1)
    inside_up = np.all(~N | up[None, :], axis=1)
    low_p = inA & inside_up
    up_p = inA | (~inA & hits_low)
    low_s = inA & hits_low
    up_s = inA | (up & inside_up)
    return dict(low=low, up=up, low_p=low_p, up_p=up_p, low_s=low_s, up_s=up_s,
                low_g=low_p | low_s, up_g=up_p & up_s)


def gamma_near(D, y, eps):
    """near-γ bağımlılığı: |∪_k lower^γ(D_k)| / |U| (karar sınıfları ayrık ⇒ toplam)."""
    N = closure(tolerance(D, eps))
    pos = 0
    for c in np.unique(y):
        pos += near_approximations(N, y == c)["low_g"].sum()
    return float(pos / len(y))


def delta_near(D, y, eps):
    """Makale Def. 2.15 near accuracy δ^γ, karar sınıfları üzerinden ortalama (teşhis amaçlı)."""
    N = closure(tolerance(D, eps))
    vals = []
    for c in np.unique(y):
        r = near_approximations(N, y == c)
        vals.append(r["low_g"].sum() / max(1, r["up_g"].sum()))
    return float(np.mean(vals))


def gamma_mem(D, y, eps):
    """Derecelendirilmiş bağımlılık: ortalama kaba üyelik  (1/n) Σ_x |T(x)∩[x]_D| / |T(x)|  (Def. kaba üyelik, sağ komşuluk)."""
    A = tolerance(D, eps)
    same = y[:, None] == y[None, :]
    return float(((A & same).sum(1) / A.sum(1)).mean())


def gamma_memN(D, y, eps):
    """Aynı ölçüt, minimal komşuluk N(x) ile."""
    N = closure(tolerance(D, eps))
    same = y[:, None] == y[None, :]
    return float(((N & same).sum(1) / N.sum(1)).mean())


GAMMAS = {"op": gamma_op, "near": gamma_near, "tol": gamma_tol, "memT": gamma_mem, "memN": gamma_memN}


# ---------------------------------------------------------------------------
# 3. ε seçimi
# ---------------------------------------------------------------------------

def eps_grid(D, n_points=25):
    """Orijinal `_get_eps_range`: çift mesafelerin 5.–95. yüzdelikleri; tamsayı veride yuvarlanır."""
    upper = D[np.triu_indices(D.shape[0], 1)]
    q = np.percentile(upper, np.linspace(5, 95, n_points))
    if np.allclose(upper, np.round(upper)):
        c = np.unique(np.round(q).astype(int))
        return [int(e) for e in c if e > 0]
    c = np.unique(np.round(q, 6))
    return [float(e) for e in c if e > 0]


def select_eps(D, y, gamma_fn, rule="argmax_lt", cap=0.999, grid=None):
    """
    rule='argmax_lt' (makale Denk. eq:eps_star_rule): argmax γ(ε), γ(ε) < cap; hepsi ≥cap ise argmax.
         Eşitlikte ilk (en küçük) ε — pandas idxmax davranışı.
    rule='coarsest' (ablasyon): γ(ε) ≥ 0.95·max_{γ<cap} γ olan EN BÜYÜK ε (en kaba tutarlı tanecik).
    """
    grid = eps_grid(D) if grid is None else grid
    g = np.array([gamma_fn(D, y, e) for e in grid])
    valid = g < cap
    if not valid.any():
        valid = np.ones_like(valid)
    idx = np.where(valid)[0]
    if rule == "argmax_lt":
        best = idx[np.argmax(g[idx])]
    elif rule == "coarsest":
        target = 0.95 * g[idx].max()
        best = idx[g[idx] >= target].max()
    else:
        raise ValueError(rule)
    return float(grid[best]), float(g[best]), np.array(grid), g


# ---------------------------------------------------------------------------
# 4. Gamma-PFI skorları
# ---------------------------------------------------------------------------

def _col_dist(x, metric, is_nominal=False):
    if is_nominal:
        return (x[:, None] != x[None, :]).astype(float)
    d = np.abs(x[:, None] - x[None, :])
    return d * d if metric == "l2" else d


def _perm_dist(Dpart, X, i, perm, metric, nominal=None):
    """Tek sütun permütasyonundan sonra mesafe matrisi (L1/L2 artımlı, Linf yeniden).
    Sütun mesafeleri saklanmaz, anında hesaplanır ⇒ bellek O(n²), p'den bağımsız."""
    nom = nominal is not None and nominal[i]
    if metric == "l1":
        return Dpart["base"] - _col_dist(X[:, i], "l1", nom) + _col_dist(perm, "l1", nom)
    if metric == "l2":
        sq = Dpart["base"] - _col_dist(X[:, i], "l2") + _col_dist(perm, "l2")
        return np.sqrt(np.maximum(sq, 0))
    Xp = X.copy()
    Xp[:, i] = perm
    return pairwise(Xp, "linf")


def _dist_parts(X, metric, nominal=None):
    if metric == "l1":
        return {"base": pairwise(X, "l1", nominal)}
    if metric == "l2":
        return {"base": sum(_col_dist(X[:, i], "l2") for i in range(X.shape[1]))}
    return {"base": None}


def gamma_pfi_mc(X, y, eps, gamma="op", n_rep=30, rng=None, metric="l1", gamma_base=None, nominal=None):
    """
    Orijinal V1Selector döngüsü (aynı RNG sırası): her öznitelik için n_rep permütasyon,
    Δγ = γ_base − γ_perm. std ddof=0 (orijinal kodla aynı).
    Döner: mu (p,), sd (p,), sem (p,), deltas (p, n_rep)
    """
    rng = np.random.RandomState(42) if rng is None else rng
    gfn = GAMMAS[gamma]
    D = pairwise(X, metric, nominal)
    gb = gfn(D, y, eps) if gamma_base is None else gamma_base
    parts = _dist_parts(X, metric, nominal)
    p = X.shape[1]
    deltas = np.zeros((p, n_rep))
    for i in range(p):
        for r in range(n_rep):
            perm = rng.permutation(X[:, i])
            deltas[i, r] = gb - gfn(_perm_dist(parts, X, i, perm, metric, nominal), y, eps)
    mu = deltas.mean(1)
    sd = deltas.std(1)
    return mu, sd, sd / np.sqrt(n_rep), deltas, gb


def gamma_pfi_boot(X, y, eps, gamma="op", n_boot=30, rng=None, metric="l1", nominal=None):
    """
    Satır-bootstrap SE: her replikasyonda satırlar sınıf-tabakalı olarak
    yerine koyarak yeniden örneklenir, γ_base^b ve tek bir permütasyonla Δγ_f^b hesaplanır.
    SE_f = sd_b(Δγ_f^b): Δγ tahmincisinin ÖRNEKLEM (+MC) belirsizliği; n_boot artınca 0'a gitmez.
    Döner: se (p,), boot_deltas (p, n_boot)
    """
    rng = np.random.RandomState(4242) if rng is None else rng
    gfn = GAMMAS[gamma]
    n, p = X.shape
    classes = np.unique(y)
    out = np.zeros((p, n_boot))
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(np.where(y == c)[0], size=(y == c).sum(), replace=True)
                              for c in classes])
        Xb, yb = X[idx], y[idx]
        parts = _dist_parts(Xb, metric, nominal)
        Db = pairwise(Xb, metric, nominal)
        gb = gfn(Db, yb, eps)
        for i in range(p):
            perm = rng.permutation(Xb[:, i])
            out[i, b] = gb - gfn(_perm_dist(parts, Xb, i, perm, metric, nominal), yb, eps)
    return out.std(1, ddof=1), out


def deletion_importance(X, y, eps, gamma="op", metric="l1"):
    """Klasik rough-set anlamlılığı: γ_B − γ_{B∖f} (ablasyon: silme vs permütasyon)."""
    gfn = GAMMAS[gamma]
    gb = gfn(pairwise(X, metric), y, eps)
    return np.array([gb - gfn(pairwise(np.delete(X, i, axis=1), metric), y, eps)
                     for i in range(X.shape[1])])


# ---------------------------------------------------------------------------
# 5. Eleme kuralları
# ---------------------------------------------------------------------------

def tstat_rule(mu, se, kappa=1.5, zero_guard=False):
    """
    CASPER-t: t_f = (mu_f − Q1(mu)) / se_f,  seç: t_f > κ.
    zero_guard=False → makaledeki/matchk_fix koşusundaki davranış ((mu−Q1)/(se+1e-9)).
    zero_guard=True  → kök pipeline'daki yeni koruma: se≈0 ise yalnızca mu>0 seçilir.
    Boş küme ⇒ en büyük mu (güvenlik ağı).
    """
    mu = np.asarray(mu, float)
    se = np.asarray(se, float)
    q1 = float(np.percentile(mu, 25))
    if zero_guard:
        z = se <= 1e-9
        t = np.where(z, np.where(mu > 0, np.inf, -np.inf), (mu - q1) / np.where(z, 1.0, se))
    else:
        t = (mu - q1) / (se + 1e-9)
    sel = np.where(t > kappa)[0]
    if sel.size == 0:
        sel = np.array([int(np.argmax(mu))])
    return sel, t, q1


def q1_point_rule(score):
    """Orijinal Stage-7 (V1_final): hepsi >0 ise Q1 üstü, değilse >0 olanlar."""
    s = np.asarray(score, float)
    if np.all(s > 0):
        sel = np.where(s > np.percentile(s, 25))[0]
    else:
        sel = np.where(s > 0)[0]
    return sel if sel.size else np.array([int(np.argmax(s))])


def positive_rule(score):
    s = np.asarray(score, float)
    sel = np.where(s > 0)[0]
    return sel if sel.size else np.array([int(np.argmax(s))])


def topk(score, k):
    return np.argsort(-np.asarray(score, float), kind="stable")[:k]


# ---------------------------------------------------------------------------
# 6. Uçtan uca seçici (ayrıklaştırılmış eğitim katı üzerinde)
# ---------------------------------------------------------------------------

def stratified_subsample(y, max_n, random_state=42):
    """Orijinal `_stratified_subsample` ile aynı (RandomState sırası dahil)."""
    n = len(y)
    if n <= max_n:
        return np.arange(n)
    rng = np.random.RandomState(random_state)
    frac = max_n / n
    parts = []
    for c in np.unique(y):
        ci = np.where(y == c)[0]
        take = min(len(ci), max(1, int(round(len(ci) * frac))))
        parts.append(rng.choice(ci, size=take, replace=False))
    return np.sort(np.concatenate(parts))


def _small_int(X):
    X = np.asarray(X, float)
    return bool(X.size and np.all(X == np.rint(X)) and np.abs(X).max() < 2 ** 20)


def pattern_gamma(X, y, eps, gamma="op", metric="l1"):
    """Exact gamma_op / gamma_tol / gamma_near of the table X, evaluated on its distinct
    code vectors.  Objects with equal rows have equal rows of D, T and N, so every
    quantity of the definitions is a function of (code vector, own class).  Integer
    codes make the pattern distances identical to the object-level ones."""
    X = np.asarray(X, float)
    yi, k = casper_fast._labels(y)
    U, pid = casper_fast._unique_rows(X)
    u, n = len(U), len(yi)
    cnt = np.bincount(pid * k + yi, minlength=u * k).reshape(u, k)   # objects per (pattern, class)
    mult = cnt.sum(1)
    Du = pairwise(U, metric)
    if gamma == "op":
        cls = np.where((cnt > 0).sum(1) == 1, cnt.argmax(1), -1)
        return float(mult[casper_fast._lower(Du, cls, eps + 1e-12)].sum()) / n
    A = tolerance(Du, eps)
    if gamma == "tol":
        cls = np.where((cnt > 0).sum(1) == 1, cnt.argmax(1), -1)
        ok = (cls >= 0) & np.all(~A | (cls[None, :] == cls[:, None]), axis=1)
        return float(mult[ok].sum()) / n
    if gamma == "near":
        N = closure(A)
        pos = 0
        for c in range(k):
            full = cnt[:, c] == mult          # every object of Q in class c
            some = cnt[:, c] > 0
            low = np.all(~N | full[None, :], axis=1)
            up = np.any(N & some[None, :], axis=1)
            hits_low = np.any(N & low[None, :], axis=1)
            inside_up = np.all(~N | up[None, :], axis=1)
            pos += int(cnt[:, c][inside_up | hits_low].sum())   # low_g = inA & (inside_up | hits_low)
        return float(pos / n)
    raise ValueError(gamma)


def pattern_pfi_mc(X, y, eps, gamma, n_rep, rng, metric, gamma_base):
    """Same loop and random stream as gamma_pfi_mc, gamma evaluated by pattern_gamma."""
    p = X.shape[1]
    deltas = np.zeros((p, n_rep))
    for i in range(p):
        Xp = X.copy()
        for r in range(n_rep):
            Xp[:, i] = rng.permutation(X[:, i])
            deltas[i, r] = gamma_base - pattern_gamma(Xp, y, eps, gamma, metric)
    mu = deltas.mean(1)
    sd = deltas.std(1)
    return mu, sd, sd / np.sqrt(n_rep), deltas, gamma_base


def pattern_pfi_boot(X, y, eps, gamma, n_boot, rng, metric):
    """Same loop and random stream as gamma_pfi_boot."""
    n, p = X.shape
    classes = np.unique(y)
    out = np.zeros((p, n_boot))
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(np.where(y == c)[0], size=(y == c).sum(), replace=True)
                              for c in classes])
        Xb, yb = X[idx], y[idx]
        gb = pattern_gamma(Xb, yb, eps, gamma, metric)
        Xp = Xb.copy()
        for i in range(p):
            Xp[:, i] = rng.permutation(Xb[:, i])
            out[i, b] = gb - pattern_gamma(Xp, yb, eps, gamma, metric)
            Xp[:, i] = Xb[:, i]
    return out.std(1, ddof=1), out


def _fast_gamma_op(X, y):
    """gamma_op(D, y, eps) for the fixed table X, evaluated on distinct patterns
    (casper_fast).  Same tolerance threshold eps + 1e-12 as ``tolerance``."""
    n = len(y)
    yi, k = casper_fast._labels(y)
    U, pid = casper_fast._unique_rows(np.asarray(X, float))
    if len(U) > casper_fast.DENSE_RATIO * n:
        d, cls, mult = pairwise(np.asarray(X, float), "l1"), yi, np.ones(n, dtype=np.int64)
    else:
        cls, mult = casper_fast._pattern_classes(pid, yi, len(U), k)
        d = pairwise(U, "l1")

    def g(D, y_, eps):
        return float(mult[casper_fast._lower(d, cls, eps + 1e-12)].sum()) / n
    return g


class CasperSelector:
    """
    gamma ∈ {'op','near','tol'}, se ∈ {'mc','boot'}, metric ∈ {'l1','l2','linf'},
    importance ∈ {'perm','delete'}, rule ∈ {'tstat','q1','positive'},
    eps_rule ∈ {'argmax_lt','coarsest','zero'}  ('zero' = ε=0, denklik ilişkisi ablasyonu).
    Varsayılanlar = makaledeki CASPER-t (op, mc, l1, perm, tstat, argmax_lt, κ=1.5, N=30, m=3000).
    """

    def __init__(self, gamma="op", se="mc", metric="l1", importance="perm", rule="tstat",
                 eps_rule="argmax_lt", kappa=1.5, n_rep=30, n_boot=30, max_n=3000,
                 zero_guard=False, random_state=42, nominal=None, engine="fast"):
        self.__dict__.update(locals())
        del self.__dict__["self"]

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y)
        sub = stratified_subsample(y.astype(float), self.max_n, self.random_state)
        Xs, ys = X[sub], y[sub]
        self.n_topological_ = len(sub)
        gfn = GAMMAS[self.gamma]
        nom = None if self.nominal is None else np.asarray(self.nominal, bool)
        D = pairwise(Xs, self.metric, nom)
        fast = (self.engine == "fast" and self.gamma == "op" and self.metric == "l1" and nom is None
                and self.importance == "perm" and self.eps_rule == "argmax_lt")
        # exact pattern engine for every other gamma / metric / rule on integer codes
        pat = self.engine == "fast" and nom is None and self.gamma in ("op", "tol", "near") and _small_int(Xs)
        if fast:
            # exact fast engine (casper_fast): same eps*, gamma_base, Delta_gamma replicates
            gfn = _fast_gamma_op(Xs, ys)
        elif pat:
            gfn = lambda D_, y_, e, _X=Xs: pattern_gamma(_X, y_, e, self.gamma, self.metric)
        if self.eps_rule == "zero":
            self.eps_star_, self.gamma_base_ = 0.0, gfn(D, ys, 0.0)
        else:
            self.eps_star_, self.gamma_base_, self.eps_grid_, self.gamma_curve_ = select_eps(
                D, ys, gfn, rule=self.eps_rule)
        rng = np.random.RandomState(self.random_state)
        if fast:
            _, _, deltas = casper_fast.gamma_pfi(Xs, ys, self.eps_star_ + 1e-12, self.gamma_base_,
                                                 n_rep=self.n_rep, rng=rng, return_deltas=True)
            mu, sd = deltas.mean(1), deltas.std(1)
            sem = sd / np.sqrt(self.n_rep)
            self.deltas_ = deltas
        elif pat and self.importance == "perm":
            mu, sd, sem, deltas, _ = pattern_pfi_mc(Xs, ys, self.eps_star_, self.gamma, self.n_rep, rng,
                                                     self.metric, self.gamma_base_)
            self.deltas_ = deltas
        elif pat:
            gb = pattern_gamma(Xs, ys, self.eps_star_, self.gamma, self.metric)
            mu = np.array([gb - pattern_gamma(np.delete(Xs, i, axis=1), ys, self.eps_star_, self.gamma, self.metric)
                           for i in range(Xs.shape[1])])
            sd = sem = np.zeros_like(mu)
        elif self.importance == "perm":
            mu, sd, sem, deltas, _ = gamma_pfi_mc(Xs, ys, self.eps_star_, self.gamma, self.n_rep,
                                                   rng, self.metric, gamma_base=self.gamma_base_, nominal=nom)
            self.deltas_ = deltas
        else:
            mu = deletion_importance(Xs, ys, self.eps_star_, self.gamma, self.metric)
            sd = sem = np.zeros_like(mu)
        self.mu_, self.sd_, self.sem_ = mu, sd, sem
        if self.se == "boot" and self.importance == "perm" and pat:
            self.se_, self.boot_deltas_ = pattern_pfi_boot(
                Xs, ys, self.eps_star_, self.gamma, self.n_boot,
                np.random.RandomState(self.random_state + 1), self.metric)
        elif self.se == "boot" and self.importance == "perm":
            self.se_, self.boot_deltas_ = gamma_pfi_boot(
                Xs, ys, self.eps_star_, self.gamma, self.n_boot,
                np.random.RandomState(self.random_state + 1), self.metric, nom)
        else:
            self.se_ = sem
        if self.rule == "tstat" and self.importance == "perm":
            sel, self.t_, self.q1_ = tstat_rule(mu, self.se_, self.kappa, self.zero_guard)
        elif self.rule == "positive":
            sel = positive_rule(mu)
        else:
            sel = q1_point_rule(mu)
        self.support_idx_ = np.sort(sel)
        self.ranking_ = np.argsort(-mu, kind="stable")
        return self
