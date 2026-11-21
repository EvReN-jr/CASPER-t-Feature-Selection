"""'Epsilon selection summary', 'K stability' ve 'kappa sensitivity' bölümleri için
kat içi, tam veri sonuçlarından tablo üretir.
- eps/K özeti: results/main/*_selection.csv (CASPER-t satırları) -> ek hesap yok.
- kappa taraması: saklanan mu/se'den kappa in {1.0,1.5,2.0,2.5} için seçim yeniden kurulur (kappa=1.5 saklanan
  seçimle birebir karşılaştırılır), seçilen alt kümeler aynı katlarda 8 sınıflandırıcıyla değerlendirilir.
Çıktı: results/kappa/<ds>.csv, results/tables/T_eps_summary.md, T_kappa.md, tables/tab_eps_summary.tex, tab_kappa.tex
"""
import os, sys, json
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from exp_common import (load, folds, FoldPreprocessor, evaluate, set_threads, MAIN9, RESULTS)

KAPPAS = [1.0, 1.5, 2.0, 2.5]
NAMES = {"cyron": "Cryotherapy", "diabet": "Diabetes", "kanser": "Breast Cancer", "parkinson": "Parkinson's",
         "sonar": "Sonar", "banknote": "Banknote", "phoneme": "Phoneme", "htru2": "HTRU2", "magic": "MAGIC"}
OUT = os.path.join(RESULTS, "kappa"); os.makedirs(OUT, exist_ok=True)
TEX = os.path.join(os.path.dirname(RESULTS), "tables")


def select(mu, se, kappa):
    mu, se = np.asarray(mu, float), np.asarray(se, float)
    q1 = np.percentile(mu, 25)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, (mu - q1) / se, np.where(mu > q1, np.inf, -np.inf))
    idx = [i for i in range(len(mu)) if t[i] > kappa]
    return idx if idx else [int(np.argmax(mu))]


def run_fold(ds, fi, tr, te, X, y, row):
    set_threads(2)
    pp = FoldPreprocessor().fit(X.iloc[tr])
    Xtr = pp.transform(X.iloc[tr]).reset_index(drop=True); Xte = pp.transform(X.iloc[te]).reset_index(drop=True)
    cols = list(Xtr.columns)
    mu, se = json.loads(row.mu), json.loads(row.se)
    stored = set(json.loads(row.Features))
    out, cache = [], {}
    for k in KAPPAS:
        idx = select(mu, se, k)
        names = [cols[i] for i in idx]
        if k == 1.5:
            assert set(names) == stored, (ds, fi, set(names) ^ stored)
        key = tuple(sorted(idx))
        if key not in cache:
            res, _ = evaluate(Xtr, Xte, y[tr], y[te], names)
            cache[key] = np.mean([m["ACCU"] for c, m in res.items() if c != "HYBRID"])
        out.append(dict(Dataset=ds, Fold=fi, kappa=k, K=len(idx), p=len(cols), ACCU=cache[key],
                        Features=json.dumps(names)))
    return out


def main():
    rows_eps = []
    for ds in MAIN9:
        S = pd.read_csv(os.path.join(RESULTS, "main", f"{ds}_selection.csv"))
        c = S[S.Method == "CASPER-t"].sort_values("Fold")
        rows_eps.append(dict(Dataset=NAMES[ds], p=int(c.p.iloc[0]), eps_min=c.eps.min(), eps_max=c.eps.max(),
                             K_mean=c.K.mean(), K_min=c.K.min(), K_max=c.K.max(), ret=100 * c.K.mean() / c.p.iloc[0],
                             cv=c.K.std(ddof=0) / c.K.mean(), gbase=c.gamma_base.mean()))
        f = os.path.join(OUT, f"{ds}.csv")
        if os.path.exists(f):
            continue
        X, y = load(ds)
        jobs = []
        for fi, (tr, te) in enumerate(folds(X, y), 1):
            r = c[c.Fold == fi].iloc[0]
            jobs.append(delayed(run_fold)(ds, fi, tr, te, X, y, r))
        res = Parallel(n_jobs=8)(jobs)
        pd.DataFrame([x for r in res for x in r]).to_csv(f, index=False)
        print("ok", ds, flush=True)
    E = pd.DataFrame(rows_eps)
    E.to_markdown(os.path.join(RESULTS, "tables", "T_eps_summary.md"), index=False, floatfmt=".3f")
    K = pd.concat(pd.read_csv(os.path.join(OUT, f"{d}.csv")) for d in MAIN9)
    base = K[K.kappa == 1.5].set_index(["Dataset", "Fold"]).Features.map(lambda s: set(json.loads(s)))
    K["J15"] = [len(set(json.loads(f)) & base[(d, fo)]) / len(set(json.loads(f)) | base[(d, fo)])
                for d, fo, f in zip(K.Dataset, K.Fold, K.Features)]
    agg = K.groupby("kappa").agg(K=("K", "mean"), ret=("K", lambda s: np.nan), ACCU=("ACCU", "mean"), J=("J15", "mean"))
    agg["ret"] = K.assign(r=K.K / K.p).groupby("kappa").r.mean() * 100
    agg["ACCU"] *= 100
    agg.to_markdown(os.path.join(RESULTS, "tables", "T_kappa.md"), floatfmt=".3f")
    perds = K.assign(ACCU=100 * K.ACCU).pivot_table(index="Dataset", columns="kappa", values="ACCU")
    perds.to_markdown(os.path.join(RESULTS, "tables", "T_kappa_by_dataset.md"), floatfmt=".2f")

    # LaTeX
    L = [r"% OTOMATIK URETILDI -- code/run_kappa_eps.py", r"\begin{table}[t]", r"\centering",
         r"\caption{Threshold and retention summary of \CASPERt{} ($\kappa=1.5$) over the 10 outer folds. $p$ is the number of features after the fold-confined correlation filter, $\varepsilon^{*}$ the range of selected thresholds, $K$ the number of retained features, CV$_K$ its coefficient of variation over folds. All quantities are refitted inside each training fold.}",
         r"\label{tab:eps_summary}", r"\small", r"\begin{tabular}{lrrrrrr}", r"\toprule",
         r"Dataset & $p$ & $\varepsilon^{*}$ range & $K$ mean & $K$ range & retained (\%) & CV$_K$ \\", r"\midrule"]
    for _, r in E.iterrows():
        er = f"{r.eps_min:.0f}" if r.eps_min == r.eps_max else f"{r.eps_min:.0f}--{r.eps_max:.0f}"
        kr = f"{r.K_min}" if r.K_min == r.K_max else f"{r.K_min}--{r.K_max}"
        L.append(f"{r.Dataset} & {r.p} & {er} & {r.K_mean:.1f} & {kr} & {r.ret:.0f} & {r.cv:.2f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(TEX, "tab_eps_summary.tex"), "w").write("\n".join(L) + "\n")
    L = [r"% OTOMATIK URETILDI -- code/run_kappa_eps.py", r"\begin{table}[t]", r"\centering",
         r"\caption{Sensitivity to the margin $\kappa$ on the nine main datasets. The selection is rebuilt from the stored permutation means and standard errors of every fold, and the subsets are evaluated on the same outer test folds. Mean $K$, mean retained share, classifier-averaged accuracy and mean Jaccard overlap with the default subset ($\kappa=1.5$).}",
         r"\label{tab:kappa}", r"\small", r"\begin{tabular}{lrrrr}", r"\toprule",
         r"$\kappa$ & mean $K$ & retained (\%) & accuracy (\%) & Jaccard to $\kappa=1.5$ \\", r"\midrule"]
    for k, r in agg.iterrows():
        L.append(f"{k:.1f} & {r.K:.2f} & {r.ret:.1f} & {r.ACCU:.2f} & {r.J:.2f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(TEX, "tab_kappa.tex"), "w").write("\n".join(L) + "\n")
    print(E.to_string()); print(agg.to_string())


if __name__ == "__main__":
    main()
