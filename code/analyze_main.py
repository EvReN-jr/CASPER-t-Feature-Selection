# -*- coding: utf-8 -*-
"""
Ana koşunun (results/main) özetleri → results/tables/*.md (+ LaTeX parçaları).
  T_main_candidates : 4 aday (γ_op/near × MC/boot) — öncesinden yazılmış seçim ölçütleri
  T_main_vs_all     : CASPER-t vs ALL — Wilcoxon, Nadeau–Bengio, BH-q (sızıntısız protokol)
  T_main_matched    : eşlenik-K: CASPER-t vs 9 klasik + yeni baseline'lar + Rastgele-K, Friedman/ID/CD
  T_main_own        : kendi durma kuralı olanların kendi K'sı (K/p ve doğruluk)
  T_runtime         : seçicilerin medyan duvar-saati süresi
"""
import os, json, glob
import numpy as np, pandas as pd
from exp_common import RESULTS, ROOT, MAIN9, PRETTY, nogueira
from stats_utils import nb_corrected_t, wilcoxon_p, bh, friedman_iman_davenport, bayes_signed_rank

TAB = os.path.join(RESULTS, "tables"); os.makedirs(TAB, exist_ok=True)
CANDS = ["CASPER-t", "CASPER-t[boot]", "CASPER-t[near]", "CASPER-t[near,boot]"]


def load_exp(exp, datasets):
    M, S = [], []
    for ds in datasets:
        f = os.path.join(RESULTS, exp, f"{ds}_metrics.csv")
        if os.path.exists(f):
            M.append(pd.read_csv(f)); S.append(pd.read_csv(os.path.join(RESULTS, exp, f"{ds}_selection.csv")))
        g = os.path.join(RESULTS, "recent", f"{ds}_metrics.csv")   # granular-ball ailesi (run_recent.py)
        if exp == "main" and os.path.exists(g) and os.path.getsize(g) > 5:
            M.append(pd.read_csv(g))
            s_ = pd.read_csv(os.path.join(RESULTS, "recent", f"{ds}_selection.csv"))
            S.append(s_[s_.get("status", "ok") == "ok"] if "status" in s_ else s_)
    return pd.concat(M), pd.concat(S)


def fold_avg(M, keys=("Dataset", "Fold", "Method", "Kind")):
    """Sınıflandırıcı-ortalamalı kat skorları (+ HYBRID ayrı)."""
    m = M[M.Classifier != "HYBRID"].groupby(list(keys), as_index=False)[["ACCU", "F1_S", "AUC", "K"]].mean()
    h = M[M.Classifier == "HYBRID"].groupby(list(keys), as_index=False)["ACCU"].mean().rename(columns={"ACCU": "HYB"})
    return m.merge(h, on=list(keys))


def md(df, name, floatfmt=".4f"):
    path = os.path.join(TAB, name + ".md")
    with open(path, "w") as f:
        f.write(df.to_markdown(index=False, floatfmt=floatfmt))
    return path


def candidates(F, S, dsets):
    rows = []
    allf = F[F.Method == "ALL"].set_index(["Dataset", "Fold"])
    for c in CANDS:
        sub = F[(F.Method == c) & (F.Kind == "own")].set_index(["Dataset", "Fold"])
        d = (sub.ACCU - allf.ACCU.reindex(sub.index)) * 100
        per = d.groupby(level=0).mean()
        ss = S[S.Method == c]
        stab = []
        for ds in dsets:
            s = ss[ss.Dataset == ds]
            if len(s):
                feats = sorted({f for x in s.Features for f in json.loads(x)})
                pos = {f: i for i, f in enumerate(sorted(set(feats) | set()))}
                p = int(s.p.iloc[0])
                sets = [{pos[f] for f in json.loads(x)} for x in s.Features]
                stab.append(nogueira(sets, max(p, len(pos))))
        sig = 0
        for ds in dsets:
            dd = d.xs(ds, level=0) if ds in d.index.get_level_values(0) else None
            if dd is not None and len(dd) > 1 and dd.mean() < 0 and nb_corrected_t(dd.values)[1] < 0.05:
                sig += 1
        rows.append({"Aday": c, "γ": "near" if "near" in c else "op", "SE": "boot" if "boot" in c else "MC",
                     "ortanca ΔAcc vs ALL (puan)": per.median(), "ortalama ΔAcc": per.mean(),
                     "en kötü ΔAcc": per.min(), "ort. K/p": (ss.K / ss.p).mean(),
                     "K=1 olan kat oranı": (ss.K == 1).mean(),
                     "Nogueira kararlılık (ort.)": np.nanmean(stab), "NB-anlamlı kayıp (veri sayısı)": sig})
    return pd.DataFrame(rows)


def vs_all(F, S, dsets, method="CASPER-t"):
    rows = []
    for ds in dsets:
        a = F[(F.Dataset == ds) & (F.Method == "ALL")].set_index("Fold")
        c = F[(F.Dataset == ds) & (F.Method == method) & (F.Kind == "own")].set_index("Fold")
        if c.empty:
            continue
        r = {"Veri": PRETTY.get(ds, ds), "p": int(S[(S.Dataset == ds)].p.iloc[0]),
             "K ort.": c.K.mean(), "ALL Acc": a.ACCU.mean() * 100, "CASPER-t Acc": c.ACCU.mean() * 100}
        for met in ("ACCU", "F1_S", "AUC"):
            d = (c[met] - a[met].reindex(c.index)).values
            r[f"Δ{met}"] = d.mean() * 100
            r[f"p_W({met})"] = wilcoxon_p(d)
            r[f"p_NB({met})"] = nb_corrected_t(d)[1]
        rows.append(r)
    T = pd.DataFrame(rows)
    for met in ("ACCU", "F1_S", "AUC"):
        T[f"q_NB({met})"] = bh(T[f"p_NB({met})"].values)
    return T


def matched(F, dsets, metric="ACCU"):
    mm = F[(F.Kind.isin(["matched"])) | (F.Method == "CASPER-t") & (F.Kind == "own") | (F.Method == "ALL")]
    piv = mm.groupby(["Dataset", "Method"])[metric].mean().unstack() * 100
    piv = piv.loc[[d for d in dsets if d in piv.index]]
    cols = ["ALL", "CASPER-t"] + [c for c in piv.columns if c not in ("ALL", "CASPER-t")]
    piv = piv[cols].dropna(axis=1)
    fr = friedman_iman_davenport(piv.drop(columns=["ALL"]).values)
    ranks = pd.Series(fr["mean_ranks"], index=[c for c in piv.columns if c != "ALL"])
    out = piv.copy(); out.index = [PRETTY.get(i, i) for i in out.index]
    out.loc["Ortalama sıra (ALL hariç)"] = [np.nan] + list(ranks.values)
    return out.reset_index().rename(columns={"index": "Veri"}), fr, piv


def runtime(S):
    S = S.copy()
    S["Method"] = S.Method.str.replace(r"\[own\]", "", regex=True)
    T = S.groupby(["Dataset", "Method"]).Seconds.median().unstack()
    T.index = [PRETTY.get(i, i) for i in T.index]
    return T.reset_index().rename(columns={"index": "Veri"})


def main(exp="main", dsets=MAIN9, tag="main"):
    M, S = load_exp(exp, dsets)
    F = fold_avg(M)
    out = {}
    C = candidates(F, S, dsets); md(C, f"T_{tag}_candidates"); out["candidates"] = C
    V = vs_all(F, S, dsets); md(V, f"T_{tag}_vs_all"); out["vs_all"] = V
    for met in ("ACCU", "F1_S", "AUC"):
        Mt, fr, piv = matched(F, dsets, met)
        md(Mt, f"T_{tag}_matched_{met}", ".2f")
        out[f"friedman_{met}"] = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in fr.items()}
        if met == "ACCU":
            b = {}
            for m in piv.columns:
                if m not in ("CASPER-t", "ALL"):
                    b[m] = bayes_signed_rank(piv["CASPER-t"].values / 100, piv[m].values / 100, rope=0.01)
            out["bayes_vs_casper"] = b
    md(runtime(S), f"T_{tag}_runtime", ".3f")
    json.dump({k: v for k, v in out.items() if not isinstance(v, pd.DataFrame)},
              open(os.path.join(TAB, f"{tag}_stats.json"), "w"), indent=1, default=float)
    return out


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "highdim":
        from exp_common import HIGHDIM
        main("highdim", HIGHDIM, "highdim")
    else:
        main()
    print("tamam")
