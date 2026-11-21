# -*- coding: utf-8 -*-
"""
results/ → tables/tab_*.tex (makaleye \\input edilen LaTeX tabloları). ELLE DÜZENLEMEYİN.
Hepsi booktabs; başlık tablonun üstünde; sayılar yüzde puan; aynı biçim.
"""
import os, json, glob
import numpy as np, pandas as pd
from exp_common import RESULTS, REV, MAIN9, HIGHDIM, PRETTY, nogueira
from analyze_main import load_exp, fold_avg, vs_all, candidates
from stats_utils import friedman_iman_davenport, nb_corrected_t, bh, bayes_signed_rank

OUT = os.path.join(REV, "tables"); os.makedirs(OUT, exist_ok=True)
HDR = "% OTOMATIK URETILDI -- code/make_tables.py. ELLE DUZENLEMEYIN.\n"
NAMES = {"CASPER-t": r"\CASPERt{}", "CASPER-t[boot]": r"\CASPERt{} (bootstrap SE)", "CASPER-t[near]": r"\CASPERt{} ($\gamma^{\mathrm{near}}$)",
         "CASPER-t[near,boot]": r"\CASPERt{} ($\gamma^{\mathrm{near}}$, bootstrap SE)", "Random-K": r"Random-$K$ (mean of draws)",
         "mrmr": "mRMR", "boruta": "Boruta", "hsic_lasso": "HSIC-Lasso", "frfs": "FRFS", "quickreduct": "QuickReduct",
         "nrs_hu": r"NRS (Hu, $\delta$-ball)", "gbrs": "GBRS (2025)", "gbfrs": "GBFRS (2026)", "gbnrs": "GBNRS", "StdPFI": "Std.\\ PFI", "ALL": "ALL (no selection)"}
ORDER = ["ALL", "CASPER-t", "Random-K", "ANOVA", "MI", "LASSO", "RF", "GB", "RFE", "SHAP", "ReliefF", "StdPFI",
         "mrmr", "boruta", "hsic_lasso", "frfs", "quickreduct", "nrs_hu", "gbnrs", "gbfrs", "gbrs"]
SHORT = {"cyron": "Cryo", "diabet": "Diab", "kanser": "BC", "parkinson": "Park", "sonar": "Sonar", "banknote": "Bank",
         "phoneme": "Phon", "htru2": "HTRU2", "magic": "MAGIC", "ozone": "Ozone", "hillvalley": "HillV", "mfeat": "MFeat",
         "usps": "USPS", "madeline": "Madel.", "philippine": "Phil.", "madelon": "Madelon"}


def nm(m):
    return NAMES.get(m, m)


def f(x, d=2):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def write(name, s):
    open(os.path.join(OUT, name), "w").write(HDR + s)


def table(caption, label, colspec, header, rows, size="\\small", note=None, fit=False):
    s = [r"\begin{table}[t]", r"\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}", size,
         r"\setlength{\tabcolsep}{3.5pt}"] + ([r"\resizebox{\linewidth}{!}{%"] if fit else []) + [f"\\begin{{tabular}}{{{colspec}}}", r"\toprule", header + r" \\", r"\midrule"]
    s += [r if r in (r"\midrule", r"\addlinespace") else r + r" \\" for r in rows]
    s += [r"\bottomrule", r"\end{tabular}" + ("}" if fit else "")]
    if note:
        s.append(f"\\par\\smallskip\\parbox{{0.95\\linewidth}}{{\\footnotesize {note}}}")
    s.append(r"\end{table}")
    return "\n".join(s) + "\n"


# ----------------------------------------------------------------------------- datasets
def t_datasets():
    from exp_common import load
    rows = []
    srcs = {"cyron": "UCI", "diabet": "sklearn", "kanser": "UCI", "parkinson": "UCI", "sonar": "UCI",
            "banknote": "OpenML", "phoneme": "OpenML", "htru2": "OpenML", "magic": "OpenML"}
    for grp, ds_list in (("main", MAIN9), ("highdim", HIGHDIM)):
        S = None
        try:
            _, S = load_exp(grp, ds_list)
        except Exception:
            pass
        for ds in ds_list:
            try:
                X, y = load(ds)
            except Exception:
                continue
            pf = S[S.Dataset == ds].p.median() if S is not None and len(S[S.Dataset == ds]) else np.nan
            rows.append(f"{PRETTY[ds]} & {len(y)} & {X.shape[1]} & {f(pf,0)} & {100*y.mean():.1f} & {srcs.get(ds,'OpenML')}")
    write("tab_datasets.tex", table(
        r"Datasets. $p_{\mathrm{raw}}$: features in the public source; $p$: median number after the redundancy filter, refitted on every training fold; \% pos.: share of the positive class.",
        "tab:datasets", "lrrrrl", r"Dataset & $n$ & $p_{\mathrm{raw}}$ & $p$ & \% pos. & Source", rows))


# ----------------------------------------------------------------------------- candidates
def t_candidates(F, S):
    C = candidates(F, S, MAIN9)
    rows = [f"{nm(r['Aday'])} & {f(r['ortanca ΔAcc vs ALL (puan)'])} & {f(r['ortalama ΔAcc'])} & {f(r['en kötü ΔAcc'])} & "
            f"{f(r['ort. K/p'])} & {f(100*r['K=1 olan kat oranı'],0)} & {f(r['Nogueira kararlılık (ort.)'])} & {int(r['NB-anlamlı kayıp (veri sayısı)'])}"
            for _, r in C.iterrows()]
    write("tab_candidates.tex", table(
        r"The four variants of Section \ref{sec:gpfi}--\ref{sec:tstat_rule} on the nine main datasets ($\kappa=1.5$). $\Delta$Acc: classifier-averaged accuracy minus ALL, in points (median, mean and worst dataset); $K/p$: retained fraction; $K{=}1$: \% of folds collapsing to one feature; $\hat\Phi$: Nogueira stability; last column: datasets with a significant loss (Nadeau--Bengio, $p<0.05$).",
        "tab:candidates", "lrrrrrrr", r"Variant & med.\ $\Delta$ & mean $\Delta$ & worst $\Delta$ & $K/p$ & $K{=}1$ (\%) & $\hat\Phi$ & sig.\ loss", rows))
    return C


# ----------------------------------------------------------------------------- vs ALL
def t_vs_all(F, S):
    V = vs_all(F, S, MAIN9)
    rows = []
    for _, r in V.iterrows():
        rows.append(f"{r['Veri']} & {int(r['p'])} & {f(r['K ort.'],1)} & {f(r['ALL Acc'])} & {f(r['CASPER-t Acc'])} & "
                    f"{f(r['ΔACCU'])} & {f(r['q_NB(ACCU)'],3)} & {f(r['ΔF1_S'])} & {f(r['q_NB(F1_S)'],3)} & {f(r['ΔAUC'])} & {f(r['q_NB(AUC)'],3)}")
    write("tab_all_vs_casper.tex", table(
        r"\CASPERt{} against the full feature set (ALL) under the leakage-free protocol. Classifier-averaged means over 10 folds; $\Delta$ in points; $q$: Benjamini--Hochberg adjusted Nadeau--Bengio $p$-values within each metric.",
        "tab:performance_summary", "lrrrrrrrrrr",
        r"Dataset & $p$ & $K$ & ALL & \CASPERt{} & $\Delta$Acc & $q$ & $\Delta$F1 & $q$ & $\Delta$AUC & $q$", rows))
    return V


# ----------------------------------------------------------------------------- matched-K
def t_matched(F, dsets, label, fname, caption, metric="ACCU"):
    mm = F[(F.Kind == "matched") | ((F.Method == "CASPER-t") & (F.Kind == "own")) | (F.Method == "ALL")]
    piv = mm.groupby(["Dataset", "Method"])[metric].mean().unstack() * 100
    piv = piv.loc[[d for d in dsets if d in piv.index]]
    meths = [m for m in ORDER if m in piv.columns and piv[m].notna().all()]
    piv = piv[meths]
    sel = [m for m in meths if m != "ALL"]
    fr = friedman_iman_davenport(piv[sel].values)
    rk = dict(zip(sel, fr["mean_ranks"]))
    best = piv[sel].max(axis=1)
    rows = []
    for m in meths:
        cells = []
        for ds in piv.index:
            v = piv.loc[ds, m]
            s = f(v)
            if m != "ALL" and abs(v - best[ds]) < 1e-9:
                s = r"\textbf{" + s + "}"
            cells.append(s)
        rows.append(f"{nm(m)} & " + " & ".join(cells) + f" & {f(rk.get(m, np.nan))}")
        if m in ("ALL", "Random-K"):
            rows.append(r"\midrule")
    head = "Method & " + " & ".join(SHORT[d] for d in piv.index) + r" & rank"
    note = (f"Friedman $\\chi^2={fr['chi2']:.2f}$ ($p={fr['p']:.3f}$); Iman--Davenport $F={fr['F_ID']:.2f}$ ($p={fr['p_ID']:.3f}$); "
            f"Nemenyi critical difference CD $={fr['CD']:.2f}$ ($k={fr['k']}$, $N={fr['N']}$). Bold: best selector per dataset.")
    write(fname, table(caption, label, "l" + "r" * (len(piv.index) + 1), head, rows, size="\\scriptsize", note=note))
    return piv, fr


# ----------------------------------------------------------------------------- own cardinality
def t_own(F, S):
    own = S[S.Method.str.endswith("[own]") | (S.Method == "CASPER-t")].copy()
    own["Method"] = own.Method.str.replace(r"\[own\]", "", regex=True)
    own["Kp"] = own.K / own.p
    kp = own.groupby(["Dataset", "Method"]).Kp.mean().unstack() * 100
    acc = F[(F.Kind == "own")].groupby(["Dataset", "Method"]).ACCU.mean().unstack() * 100
    meths = [m for m in ["CASPER-t", "boruta", "frfs", "quickreduct", "nrs_hu", "gbnrs", "gbfrs", "gbrs"] if m in acc.columns]
    rows = []
    for ds in [d for d in MAIN9 if d in acc.index]:
        rows.append(PRETTY[ds] + " & " + " & ".join(f"{f(acc.loc[ds,m])} ({f(kp.loc[ds,m],0)})" for m in meths))
    write("tab_own.tex", table(
        r"Methods with their own stopping rule, each at its own cardinality: classifier-averaged accuracy (\%) and, in parentheses, the retained share $K/p$ (\%).",
        "tab:own", "l" + "r" * len(meths), "Dataset & " + " & ".join(nm(m) for m in meths), rows, size="\\small", fit=True))


# ----------------------------------------------------------------------------- runtime
def t_runtime(S, dsets, fname="tab_runtime.tex", label="tab:runtime", machine=r"same machine (16-core CPU, 30\,GB RAM)"):
    S = S.copy(); S["Method"] = S.Method.str.replace(r"\[own\]", "", regex=True)
    S = S.drop_duplicates(["Dataset", "Fold", "Method"])
    T = S.groupby(["Method", "Dataset"]).Seconds.median().unstack()
    T = T[[d for d in dsets if d in T.columns]]
    meths = [m for m in ORDER + ["CASPER-t[boot]", "CASPER-t[near]", "CASPER-t[near,boot]"] if m in T.index]
    rows = [f"{nm(m)} & " + " & ".join(f(T.loc[m, d], 2 if T.loc[m, d] < 10 else 0) for d in T.columns) for m in meths]
    write(fname, table(
        r"Median wall-clock time (seconds) of one selector fit on one training fold, " + machine + r", all methods single-process. \CASPERt{} includes discretization and the $\varepsilon$ scan.",
        label, "l" + "r" * len(T.columns), "Method & " + " & ".join(SHORT[d] for d in T.columns), rows, size="\\scriptsize"))


# ----------------------------------------------------------------------------- simple md→tex for rest
def t_from_md(mdname, fname, caption, label, cols=None, rename=None, digits=None, size="\\small"):
    p = os.path.join(RESULTS, "tables", mdname + ".md")
    if not os.path.exists(p):
        return
    lines = [l for l in open(p).read().splitlines() if l.startswith("|")]
    head = [c.strip() for c in lines[0].strip("|").split("|")]
    data = [[c.strip() for c in l.strip("|").split("|")] for l in lines[2:]]
    df = pd.DataFrame(data, columns=head)
    if cols:
        df = df[cols]
    if rename:
        df = df.rename(columns=rename)
    rows = [" & ".join(str(v).replace("_", r"\_").replace("%", r"\%") for v in r) for r in df.values]
    write(fname, table(caption, label, "l" + "r" * (df.shape[1] - 1), " & ".join(df.columns), rows, size=size))


# ----------------------------------------------------------------------------- ek deneyler
def _md(name):
    p = os.path.join(RESULTS, "tables", name + ".md")
    if not os.path.exists(p):
        return None
    lines = [l for l in open(p).read().splitlines() if l.startswith("|")]
    head = [c.strip() for c in lines[0].strip("|").split("|")]
    df = pd.DataFrame([[c.strip() for c in l.strip("|").split("|")] for l in lines[2:]], columns=head)
    for c in df.columns:
        try:
            df[c] = df[c].astype(float)
        except ValueError:
            pass
    return df


ABL = {"REF": "Reference (Jenks-KMeans, $L_1$, $N(x)$, permutation, margin rule)",
       "disc=jenks_exact": "Discretization: exact Fisher--Jenks (DP)", "disc=equal_width": "Discretization: equal width",
       "disc=equal_freq": "Discretization: equal frequency", "disc=none(raw)": "Discretization: none (rescaled raw)",
       "metric=L2": "Distance: $L_2$", "metric=Linf": r"Distance: $L_\infty$",
       "relation=equivalence(eps=0)": r"Relation: equivalence ($\varepsilon=0$)",
       "eps_rule=coarsest": r"$\varepsilon$ rule: coarsest consistent", "nbhd=ball T(x) (Hu)": r"Neighbourhood: $\delta$-ball $T(x)$",
       "nbhd=near-gamma": r"Dependency: $\gamma^{\mathrm{near}}$", "importance=deletion": "Importance: deletion $\\gamma_B-\\gamma_{B\\setminus f}$",
       "rule=Q1 point estimate": "Rule: $\\hat\\mu_f>Q_1$ (no margin)", "rule=positive mean": "Rule: $\\hat\\mu_f>0$"}


def t_ablation():
    d = _md("T_ablation")
    if d is None:
        return
    rows = []
    for _, r in d.iterrows():
        rows.append(f"{ABL.get(r['Varyant'], r['Varyant'])} & {f(r['ort. K/Kref'])} & {f(r['ΔAcc own (ort.)'])} & {f(r['ΔAcc own (en kötü)'])} & {int(r['anlamlı kayıp (own)'])}"
                    f" & {f(r['ΔAcc matched (ort.)'])} & {f(r['ΔAcc matched (en kötü)'])} & {int(r['anlamlı kayıp (matched)'])} & {f(r['süre (s, medyan)'])}")
    write("tab_ablation.tex", table(
        r"Component-wise ablation on the nine main datasets (one component changed at a time). $K/K_{\mathrm{ref}}$: mean ratio of retained features; $\Delta$Acc: accuracy difference to the reference (points), mean and worst dataset, at the variant's own $K$ and at the reference $K$ (ranking only); sig.: datasets with a significant loss (Nadeau--Bengio, $p<0.05$); time: median seconds per fold.",
        "tab:ablation", "lrrrrrrrr", r"Variant & $K/K_{\mathrm{ref}}$ & \multicolumn{3}{c}{own $K$: mean / worst / sig.} & \multicolumn{3}{c}{matched $K$: mean / worst / sig.} & time",
        rows, size="\\scriptsize"))


def t_nperm():
    d = _md("T_nperm")
    if d is None:
        return
    d = d[(d.SE == "mc") | (d.Ayar.isin(["BOOT_B30_k1.5", "BOOT_B30_k0.5", "BOOT_B30_k0.25"]))]
    rows = []
    for _, r in d.iterrows():
        lab = (f"MC, $N={int(r.N)}$, $\\kappa={r.kappa:g}$" if r.SE == "mc" else f"bootstrap, $B={int(r.B)}$, $\\kappa={r.kappa:g}$")
        rows.append(f"{lab} & {f(r['ort. K'])} & {f(r['Acc (sınıfl. ort.)'])} & {f(r['Jaccard vs MC-N200'])} & {f(r['medyan abs(t)'])}")
    write("tab_nperm.tex", table(
        r"Sensitivity to the number of shuffles $N$ and to the standard error used in the margin (nine main datasets, 10 folds each). Jaccard: overlap with the subset selected at $N=200$; median $\lvert t\rvert$: median margin over features.",
        "tab:nperm", "lrrrr", r"Setting & mean $K$ & Acc (\%) & Jaccard vs $N{=}200$ & median $|t|$", rows))


def t_kcurve():
    d = _md("T_kcurve_auc")
    if d is None:
        return
    cols = ["CASPER-t", "Random", "ANOVA", "RF", "mRMR", "NRS(Hu)"]
    rows = [f"{r['Veri']} & " + " & ".join(f(r[c]) for c in cols) + f" & {f(r['CASPER−Random'])}" for _, r in d.iterrows()]
    write("tab_kcurve.tex", table(
        r"Area under the accuracy--$K$ curve ($K=1,\dots,p$; normalized, \%) of each ranking; last column: \CASPERt{} minus random ordering.",
        "tab:kcurve", "lrrrrrrr", r"Dataset & \CASPERt{} & Random & ANOVA & RF & mRMR & NRS & $\Delta_{\mathrm{rand}}$", rows))


def t_subsample():
    d, s = _md("T_subsample"), _md("T_scaling")
    if d is not None:
        rows = [f"{int(r.m)} & {f(r['ort. K'])} & {f(r['Jaccard (tohumlar arası)'])} & {f(r['Jaccard (m=max\'a)'])} & {f(r.Acc)} & {f(r['süre (s, medyan)'],1)}" for _, r in d.iterrows()]
        write("tab_subsample.tex", table(
            r"Effect of the subsample size $m$ of the topological stage (MAGIC, 3 folds, 2 subsample seeds). Jaccard between seeds and with the $m=5000$ subset; time: seconds per fit.",
            "tab:subsample", "rrrrrr", r"$m$ & mean $K$ & Jaccard (seeds) & Jaccard ($m{=}5000$) & Acc (\%) & time (s)", rows))
    if s is not None:
        ns = [c for c in s.columns if c.startswith("n=")]
        nmz = {"CASPER-t full fit": r"full \CASPERt{} fit", "gamma_op": r"one $\gamma$ evaluation", "gamma_near": r"one $\gamma^{\mathrm{near}}$ evaluation"}
        rows = [f"{nmz.get(r['işlem'], r['işlem'])} & " + " & ".join(f(r[c], 3 if r[c] < 1 else 1) for c in ns) + f" & {f(r['log-log eğim (n≥1000)'])}" for _, r in s.iterrows()]
        write("tab_scaling.tex", table(r"Measured wall-clock time (seconds) against the number of objects $n$ (MAGIC, 16 BLAS threads) and fitted log--log slope for $n\ge1000$.",
              "tab:scaling", "l" + "r" * (len(ns) + 1), "Operation & " + " & ".join(f"${c}$" for c in ns) + " & slope", rows, size="\\scriptsize"))


def t_mixed():
    d, l = _md("T_mixed"), _md("T_labelnoise")
    if d is not None:
        rows = [f"{r.Dataset} & {nm(r.Method).replace('CASPER-t[HEOM]', r'\CASPERt{} (HEOM)')} & {f(r.Acc)} & {f(r.F1)} & {f(r.AUC)} & {f(r.K,1)}" for _, r in d.iterrows()]
        write("tab_mixed.tex", table(r"Mixed-type data with missing values (OpenML): \CASPERt{} with a Hamming term for nominal attributes (HEOM) against classical selectors at the same $K$.",
              "tab:mixed", "llrrrr", r"Dataset & Method & Acc (\%) & F1 (\%) & AUC (\%) & $K$", rows, size="\\scriptsize"))
    if l is not None:
        rows = [f"{r.Method.replace('CASPER-t', r'\CASPERt{}')} & {r.flip:.0%} & {f(r.Acc)} & {f(r.Jaccard)}".replace("%", r"\%") for _, r in l.iterrows()]
        write("tab_labelnoise.tex", table(r"Label noise injected into the training folds (six datasets): accuracy on clean test folds and Jaccard overlap with the subset selected on clean labels.",
              "tab:labelnoise", "lrrr", r"Method & flipped labels & Acc (\%) & Jaccard vs clean", rows))


def main():
    M, S = load_exp("main", MAIN9)
    F = fold_avg(M)
    t_datasets()
    t_candidates(F, S)
    t_vs_all(F, S)
    for met, lab in (("ACCU", "Accuracy"), ("F1_S", "F1"), ("AUC", "AUC")):
        t_matched(F, MAIN9, f"tab:matched_{met}", f"tab_matched_{met}.tex",
                  "Matched cardinality on the nine main datasets: classifier-averaged " + lab + r" (\%) of every selector truncated at \CASPERt{}'s per-fold $K$. Rank: mean Friedman rank among selectors (ALL excluded).", met)
    t_own(F, S)
    t_runtime(S, MAIN9)
    for fn in (t_ablation, t_nperm, t_kcurve, t_subsample, t_mixed):
        try:
            fn()
        except Exception as e:
            print("HATA", fn.__name__, e)
    try:
        Mh, Sh = load_exp("highdim", HIGHDIM)
        Fh = fold_avg(Mh)
        t_matched(Fh, HIGHDIM, "tab:master_highdim", "tab_matched_highdim.tex",
                  r"High-dimensional datasets ($p=52$--$490$ after the redundancy filter): classifier-averaged Accuracy (\%) of each ranking at \CASPERt{}'s per-fold $K$. Rank: mean Friedman rank (ALL excluded).")
        t_runtime(Sh, HIGHDIM, "tab_runtime_highdim.tex", "tab:runtime_highdim", machine=r"one node of a computing cluster (20 cores per job), topological stage on at most 1500 objects")
    except Exception as e:
        print("highdim yok:", e)
    print("tablolar yazıldı:", sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
