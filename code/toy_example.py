# -*- coding: utf-8 -*-
"""
Oyuncak örnek: 8 nesne, 3 öznitelik. CASPER-t'nin her aşaması adım adım:
Jenks kodları → L1 mesafeleri → ε taraması → T(x), N(x) → γ_op ve near-γ → her öznitelik için
TAM permütasyon dağılımı (8! = 40320) → N=30 MC tahmini → t_f → seçim.
Çıktı: results/toy/toy_example.md ve manuscript'e girecek LaTeX tabloları (results/toy/*.tex).
Testler (code/tests/test_toy.py) bu dosyadaki sayıları tanımdan bağımsız yeniden hesaplar.
"""
import os, itertools, json
import numpy as np
from casper_core import (pairwise, tolerance, closure, closure_naive, gamma_op, gamma_near, eps_grid, select_eps,
                         gamma_pfi_mc, tstat_rule)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "toy")

# Jenks sonrası sıralı kodlar (a: sınıfla ilişkili, b: kısmen, c: gürültü), ve karar sınıfı
CODES = np.array([
    # a  b  c      (arama ile bulundu: a sınıfla ilişkili, b zayıf, c gürültü)
    [1, 1, 3],
    [1, 3, 2],
    [1, 3, 3],
    [2, 3, 1],
    [3, 3, 3],
    [3, 2, 2],
    [2, 2, 2],
    [3, 3, 3],
], float)
Y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
NAMES = ["a", "b", "c"]


def exact_perm_moments(X, y, eps, gb):
    """Her öznitelik için 8! permütasyonun TAMAMI üzerinden E[Δγ] ve SD[Δγ]."""
    out = []
    n = len(y)
    for i in range(X.shape[1]):
        vals = []
        for perm in itertools.permutations(range(n)):
            Xp = X.copy(); Xp[:, i] = X[list(perm), i]
            vals.append(gb - gamma_op(pairwise(Xp), y, eps))
        vals = np.array(vals)
        out.append((vals.mean(), vals.std()))
    return np.array(out)


def run():
    os.makedirs(OUT, exist_ok=True)
    X, y = CODES, Y
    D = pairwise(X)
    grid = list(range(0, int(D.max()) + 1))
    scan = [(e, gamma_op(D, y, e), gamma_near(D, y, e)) for e in grid]
    eps, gb, _, _ = select_eps(D, y, gamma_op, grid=[e for e in grid if e > 0])
    A = tolerance(D, eps); N = closure(A)
    assert np.array_equal(N, closure_naive(A))
    mu, sd, sem, deltas, _ = gamma_pfi_mc(X, y, eps, "op", 30, np.random.RandomState(42), gamma_base=gb)
    sel, t, q1 = tstat_rule(mu, sem, 1.5)
    exact = exact_perm_moments(X, y, eps, gb)
    res = dict(eps_star=eps, gamma_base=gb, scan=scan, T=[np.where(A[i])[0].tolist() for i in range(8)],
               N=[np.where(N[i])[0].tolist() for i in range(8)],
               pos=[bool(np.all(y[N[i]] == y[i])) for i in range(8)],
               mu=mu.tolist(), sd=sd.tolist(), sem=sem.tolist(), q1=q1, t=t.tolist(),
               selected=[NAMES[i] for i in sel], exact_mean=exact[:, 0].tolist(), exact_sd=exact[:, 1].tolist(),
               D=D.astype(int).tolist())
    json.dump(res, open(os.path.join(OUT, "toy.json"), "w"), indent=1)
    # ---- Markdown
    L = ["# Oyuncak örnek — CASPER-t adım adım", "", "## 1. Jenks kodları ve sınıf", "",
         "| x | a | b | c | sınıf |", "|---|---|---|---|---|"]
    L += [f"| x{i+1} | {int(X[i,0])} | {int(X[i,1])} | {int(X[i,2])} | {y[i]} |" for i in range(8)]
    L += ["", "## 2. L1 mesafe matrisi", "", "| | " + " | ".join(f"x{j+1}" for j in range(8)) + " |",
          "|" + "---|" * 9]
    L += [f"| x{i+1} | " + " | ".join(str(int(v)) for v in D[i]) + " |" for i in range(8)]
    L += ["", "## 3. ε taraması (γ_op = kodun γ'sı; near-γ = Def. 2.14 pre∪semi)", "", "| ε | γ_op | near-γ |", "|---|---|---|"]
    L += [f"| {e} | {g:.3f} | {h:.3f} |" for e, g, h in scan]
    L += ["", f"Kural: ε* = argmax γ_op(ε), γ_op<0.999, ε>0 → **ε* = {eps:g}**, γ_base = {gb:.3f}.",
          "Not: ε=0'da γ_op=1 (önemsiz ince plato) ve ε ≥ diam'da γ_op=0, near-γ=1 (çöküş).", "",
          "## 4. Tolerans sınıfları T(x) ve minimal komşuluklar N(x) = {z : T(x) ⊆ T(z)}", "",
          "| x | T(x) | N(x) | N(x) ⊆ [x]_D ? |", "|---|---|---|---|"]
    L += [f"| x{i+1} | {{{', '.join('x'+str(j+1) for j in res['T'][i])}}} | {{{', '.join('x'+str(j+1) for j in res['N'][i])}}} | {'evet' if res['pos'][i] else 'hayır'} |" for i in range(8)]
    L += ["", f"γ_base = |POS|/|U| = {sum(res['pos'])}/8 = {gb:.3f}", "",
          "## 5. Permütasyon önemi (N=30 MC) ve tam permütasyon momentleri (8! = 40320)", "",
          "| öznitelik | μ̂ (N=30) | SD | SEM | E[Δγ] (tam) | SD (tam) | t_f | seçildi |", "|---|---|---|---|---|---|---|---|"]
    L += [f"| {NAMES[i]} | {mu[i]:.4f} | {sd[i]:.4f} | {sem[i]:.4f} | {exact[i,0]:.4f} | {exact[i,1]:.4f} | {t[i]:.2f} | {'✓' if NAMES[i] in res['selected'] else '✗'} |" for i in range(3)]
    L += ["", f"Q1(μ̂) = {q1:.4f}, κ = 1.5 → seçilen: **{', '.join(res['selected'])}**", ""]
    open(os.path.join(OUT, "toy_example.md"), "w").write("\n".join(L))
    # ---- LaTeX (makale)
    tex = [r"\begin{table}[H]", r"\centering",
           r"\caption{Toy example: Jenks codes, tolerance classes $T_{\varepsilon^*}(x)$, minimal neighbourhoods $N(x)$ and positive-region membership at $\varepsilon^*=" + f"{eps:g}" + r"$.}",
           r"\label{tab:toy_steps}", r"\begin{tabular}{lccccll c}", r"\toprule",
           r"$x$ & $a$ & $b$ & $c$ & $d$ & $T_{\varepsilon^*}(x)$ & $N(x)$ & $x\in\mathrm{POS}$ \\", r"\midrule"]
    for i in range(8):
        tx = r"\{" + ",".join(f"x_{{{j+1}}}" for j in res["T"][i]) + r"\}"
        nx = r"\{" + ",".join(f"x_{{{j+1}}}" for j in res["N"][i]) + r"\}"
        tex.append(f"$x_{{{i+1}}}$ & {int(X[i,0])} & {int(X[i,1])} & {int(X[i,2])} & {y[i]} & ${tx}$ & ${nx}$ & {'yes' if res['pos'][i] else 'no'} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}", "",
            r"\begin{table}[H]", r"\centering",
            r"\caption{Toy example: permutation importance with $N=30$ Monte Carlo replicates, the exact permutation moments over all $8!$ orderings, the margin statistic $t_f$ and the decision at $\kappa=1.5$ ($Q_1=" + f"{q1:.4f}" + r"$).}",
            r"\label{tab:toy_scores}", r"\begin{tabular}{lrrrrrrc}", r"\toprule",
            r"Feature & $\hat\mu_{\Delta\gamma_f}$ & $\hat\sigma$ & SEM & $E_\pi[\Delta\gamma_f]$ & $SD_\pi$ & $t_f$ & kept \\", r"\midrule"]
    for i in range(3):
        tex.append(f"${NAMES[i]}$ & {mu[i]:.4f} & {sd[i]:.4f} & {sem[i]:.4f} & {exact[i,0]:.4f} & {exact[i,1]:.4f} & {t[i]:.2f} & {'yes' if NAMES[i] in res['selected'] else 'no'} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(OUT, "toy_tables.tex"), "w").write("\n".join(tex))
    return res


if __name__ == "__main__":
    r = run()
    print(json.dumps({k: r[k] for k in ("eps_star", "gamma_base", "selected", "mu", "exact_mean", "t", "q1")}, indent=1))
