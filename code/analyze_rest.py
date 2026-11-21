# -*- coding: utf-8 -*-
"""
Ana koşu dışındaki deneylerin özetleri → results/tables/*.md  (+ json)
  nperm, ablation, synthetic, kcurve, subsample(+scaling), mixed, labelnoise, zero-SEM kontrolü
"""
import os, json, glob
import numpy as np, pandas as pd
from exp_common import RESULTS, PRETTY, jaccard
from stats_utils import nb_corrected_t, bh, friedman_iman_davenport

TAB = os.path.join(RESULTS, "tables"); os.makedirs(TAB, exist_ok=True)


def md(df, name, ff=".4f"):
    open(os.path.join(TAB, name + ".md"), "w").write(df.to_markdown(index=False, floatfmt=ff))


def cat(pattern):
    fs = sorted(glob.glob(os.path.join(RESULTS, pattern)))
    return pd.concat([pd.read_csv(f) for f in fs]) if fs else pd.DataFrame()


def clf_avg(M, keys):
    return M[M.Classifier != "HYBRID"].groupby(keys, as_index=False)[["ACCU", "F1_S", "AUC"]].mean()


# ------------------------------------------------------------------ nperm
def nperm():
    S, M = cat("nperm/*_selection.csv"), cat("nperm/*_metrics.csv")
    if S.empty:
        return
    t = S.merge(clf_avg(M, ["Dataset", "Fold", "Setting"]), on=["Dataset", "Fold", "Setting"])
    t["Kp"] = t.K / t.groupby(["Dataset"]).K.transform(lambda x: np.nan).fillna(1)
    ref = S[S.Setting == "MC_N200_k1.5"].set_index(["Dataset", "Fold"]).Features
    def jac(row):
        r = ref.get((row.Dataset, row.Fold))
        return np.nan if r is None else jaccard(json.loads(row.Features), json.loads(r))
    t["Jaccard_vs_N200"] = t.apply(jac, axis=1)
    rows = []
    for st, g in t.groupby("Setting"):
        rows.append(dict(Ayar=st, SE=g.SE.iloc[0], N=g.N.iloc[0], B=g.get("B", pd.Series([np.nan])).iloc[0],
                         kappa=g.kappa.iloc[0], **{"ort. K": g.K.mean(), "Acc (sınıfl. ort.)": g.ACCU.mean() * 100,
                         "Jaccard vs MC-N200": g.Jaccard_vs_N200.mean(), "medyan abs(t)": g.mean_abs_t.median()}))
    T = pd.DataFrame(rows).sort_values(["SE", "N", "B", "kappa"])
    md(T, "T_nperm", ".3f")
    per = t[t.SE == "mc"].pivot_table(index="Dataset", columns="N", values="K", aggfunc="mean")
    per.index = [PRETTY.get(i, i) for i in per.index]
    md(per.reset_index().rename(columns={"index": "Veri"}), "T_nperm_K_by_dataset", ".2f")
    return T


# ------------------------------------------------------------------ ablation
def ablation():
    S, M = cat("ablation/*_selection.csv"), cat("ablation/*_metrics.csv")
    if S.empty:
        return
    A = clf_avg(M, ["Dataset", "Fold", "Variant", "Kind"])
    ref = A[A.Variant == "REF"].set_index(["Dataset", "Fold", "Kind"]).ACCU
    rows = []
    for v in S.Variant.unique():
        r = {"Varyant": v}
        sv = S[S.Variant == v]
        r["ort. K/Kref"] = (sv.K / sv.Kref).mean()
        r["süre (s, medyan)"] = sv.Seconds.median()
        for kind in ("own", "matched"):
            a = A[(A.Variant == v) & (A.Kind == kind)].set_index(["Dataset", "Fold", "Kind"]).ACCU
            d = (a - ref.reindex(a.index)) * 100
            per = d.groupby(level=0).mean()
            r[f"ΔAcc {kind} (ort.)"] = per.mean()
            r[f"ΔAcc {kind} (en kötü)"] = per.min()
            sig = sum(1 for ds, dd in d.groupby(level=0) if len(dd) > 2 and nb_corrected_t(dd.values)[1] < 0.05 and dd.mean() < 0)
            r[f"anlamlı kayıp ({kind})"] = sig
        rows.append(r)
    T = pd.DataFrame(rows)
    md(T, "T_ablation", ".3f")
    per = A[A.Kind == "own"].pivot_table(index="Variant", columns="Dataset", values="ACCU", aggfunc="mean") * 100
    md(per.reset_index(), "T_ablation_by_dataset", ".2f")
    return T


# ------------------------------------------------------------------ synthetic
def synthetic():
    f = os.path.join(RESULTS, "synthetic", "synthetic.csv")
    if not os.path.exists(f):
        return
    D = pd.read_csv(f)
    g = D[D.flip == 0].groupby("p").agg({c: "mean" for c in D.columns if c.startswith("AUC_") or c in
                                          ("mu_inf", "mu_noise", "sem_inf", "eps_over_p", "K", "recall_true",
                                           "precision_true", "acc_casper", "acc_true5", "acc_all", "seconds")}).reset_index()
    md(g, "T_synthetic", ".3f")
    n = D[D.p == 20].groupby("flip").agg({c: "mean" for c in D.columns if c.startswith("AUC_") or c in
                                           ("recall_true", "precision_true", "K", "acc_casper")}).reset_index()
    md(n, "T_synthetic_noise", ".3f")
    # log-log eğim: mu_inf ~ p^a
    a = np.polyfit(np.log(g.p), np.log(g.mu_inf), 1)[0]
    json.dump({"slope_log_mu_inf_vs_log_p": a}, open(os.path.join(TAB, "synthetic_stats.json"), "w"))
    return g


# ------------------------------------------------------------------ kcurve
def kcurve():
    D = cat("kcurve/*.csv")
    if D.empty:
        return
    rows = []
    for (ds, m), g in D.groupby(["Dataset", "Method"]):
        c = g.groupby("K").ACCU.mean()
        rows.append(dict(Dataset=ds, Method=m, AUC_K=np.trapezoid(c.values, c.index) / (c.index.max() - c.index.min() + 1e-9),
                         acc_at_Kcasper=g[g.K == g.Kcasper.round().astype(int)].ACCU.mean()))
    T = pd.DataFrame(rows)
    piv = T.pivot(index="Dataset", columns="Method", values="AUC_K") * 100
    piv["CASPER−Random"] = piv["CASPER-t"] - piv["Random"]
    piv.index = [PRETTY.get(i, i) for i in piv.index]
    md(piv.reset_index().rename(columns={"index": "Veri"}), "T_kcurve_auc", ".2f")
    D.to_csv(os.path.join(TAB, "kcurve_long.csv"), index=False)
    return piv


# ------------------------------------------------------------------ subsample
def subsample():
    S, M = cat("subsample/*_selection.csv"), cat("subsample/*_metrics.csv")
    rows = []
    if not S.empty:
        A = clf_avg(M, ["Dataset", "Fold", "m", "seed"])
        t = S.merge(A, on=["Dataset", "Fold", "m", "seed"])
        for (ds, m), g in t.groupby(["Dataset", "m"]):
            js, jref = [], []
            for fo, gf in g.groupby("Fold"):
                sets = [json.loads(x) for x in gf.Features]
                js += [jaccard(sets[i], sets[j]) for i in range(len(sets)) for j in range(i + 1, len(sets))]
                big = t[(t.Dataset == ds) & (t.Fold == fo) & (t.m == t.m.max())].Features
                jref += [jaccard(s, json.loads(b)) for s in sets for b in big]
            rows.append({"Veri": PRETTY.get(ds, ds), "m": m, "ort. K": g.K.mean(), "Jaccard (tohumlar arası)": np.mean(js),
                         "Jaccard (m=max'a)": np.mean(jref), "Acc": g.ACCU.mean() * 100, "süre (s, medyan)": g.Seconds.median()})
        md(pd.DataFrame(rows), "T_subsample", ".3f")
    f = os.path.join(RESULTS, "subsample", "scaling_magic.csv")
    if os.path.exists(f):
        sc = pd.read_csv(f)
        out = []
        for w, g in sc.groupby("what"):
            g = g[g.n >= 1000]
            out.append({"işlem": w, "log-log eğim (n≥1000)": np.polyfit(np.log(g.n), np.log(g.seconds), 1)[0],
                        **{f"n={int(n)}": s for n, s in zip(sc[sc.what == w].n, sc[sc.what == w].seconds)}})
        md(pd.DataFrame(out), "T_scaling", ".3f")


# ------------------------------------------------------------------ mixed + labelnoise
def mixed_noise():
    M = cat("mixed/*.csv")
    if not M.empty:
        A = M[M.Classifier != "HYBRID"].groupby(["Dataset", "Fold", "Method"], as_index=False)[["ACCU", "F1_S", "AUC", "K"]].mean()
        T = A.groupby(["Dataset", "Method"]).agg(Acc=("ACCU", "mean"), F1=("F1_S", "mean"), AUC=("AUC", "mean"), K=("K", "mean")).reset_index()
        T[["Acc", "F1", "AUC"]] *= 100
        md(T, "T_mixed", ".2f")
    L = cat("labelnoise/*.csv")
    if not L.empty:
        A = L[L.Classifier != "HYBRID"].groupby(["Dataset", "Fold", "Method", "flip"], as_index=False)[["ACCU", "Jaccard_vs_clean"]].mean()
        T = A.groupby(["Method", "flip"]).agg(Acc=("ACCU", "mean"), Jaccard=("Jaccard_vs_clean", "mean")).reset_index()
        T["Acc"] *= 100
        md(T, "T_labelnoise", ".3f")


# ------------------------------------------------------------------ sıfır-SEM kontrolü
def zero_guard_check():
    from casper_core import tstat_rule
    S = cat("main/*_selection.csv")
    S = S[S.Method == "CASPER-t"]
    rows = []
    for _, r in S.iterrows():
        mu, se = np.array(json.loads(r.mu)), np.array(json.loads(r.se))
        a = set(tstat_rule(mu, se, 1.5, zero_guard=False)[0]); b = set(tstat_rule(mu, se, 1.5, zero_guard=True)[0])
        rows.append(dict(Dataset=r.Dataset, Fold=r.Fold, n_zero_se=int((se <= 1e-9).sum()), same=a == b,
                         K_noguard=len(a), K_guard=len(b)))
    T = pd.DataFrame(rows).groupby("Dataset").agg(katlar=("Fold", "count"), sıfır_SE_öznitelik_ort=("n_zero_se", "mean"),
                                                   aynı_küme_oranı=("same", "mean"), K_korumasız=("K_noguard", "mean"),
                                                   K_korumalı=("K_guard", "mean")).reset_index()
    md(T, "T_zero_guard", ".3f")
    return T


if __name__ == "__main__":
    for fn in (nperm, ablation, synthetic, kcurve, subsample, mixed_noise, zero_guard_check):
        try:
            fn(); print("ok", fn.__name__)
        except Exception as e:
            print("HATA", fn.__name__, e)
