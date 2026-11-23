"""Wall-clock time of one CASPER-t fit per outer training fold (exact evaluation, one thread per fit).
Fits run one after another, after one warm-up fit per fold that removes the numba compilation time.
The timed part includes the Jenks discretization and the epsilon scan, as in the article.
Each row also records whether eps*, gamma_base and the subset equal the stored main run (results/main).
Usage: python run_runtime.py [dataset ...]      Output: results/runtime/casper_t_runtime.csv"""
import os, sys, json, time, platform
import numpy as np, pandas as pd
from exp_common import load, folds, FoldPreprocessor, Discretizer, set_threads, RESULTS, MAIN9
from casper_core import CasperSelector

OUT = os.path.join(RESULTS, "runtime"); os.makedirs(OUT, exist_ok=True)


def main(datasets):
    set_threads(1)
    rows = []
    for ds in datasets:
        X, y = load(ds)
        S = pd.read_csv(os.path.join(RESULTS, "main", f"{ds}_selection.csv"))
        c = S[S.Method == "CASPER-t"].set_index("Fold")
        for fi, (tr, _) in enumerate(folds(X, y), 1):
            Xtr = FoldPreprocessor().fit(X.iloc[tr]).transform(X.iloc[tr]).reset_index(drop=True)
            cols = list(Xtr.columns)
            CasperSelector(max_n=3000).fit(Discretizer().fit(Xtr).transform(Xtr).values.astype(float), y[tr])  # warm-up
            t = time.perf_counter()
            Xd = Discretizer().fit(Xtr).transform(Xtr).values.astype(float)
            cs = CasperSelector(max_n=3000).fit(Xd, y[tr])
            sec = time.perf_counter() - t
            sel = sorted(cols[i] for i in cs.support_idx_)
            same = (abs(cs.eps_star_ - c.loc[fi, "eps"]) < 1e-9 and abs(cs.gamma_base_ - c.loc[fi, "gamma_base"]) < 1e-6
                    and sel == sorted(json.loads(c.loc[fi, "Features"])))
            rows.append(dict(Dataset=ds, Fold=fi, Seconds=round(sec, 4), K=len(sel), eps=cs.eps_star_, same_as_main=bool(same)))
        d = pd.DataFrame([r for r in rows if r["Dataset"] == ds])
        print(f"{ds}: median {d.Seconds.median():.3f} s, same as main {d.same_as_main.sum()}/10", flush=True)
    R = pd.DataFrame(rows)
    R["machine"] = f"{platform.processor()} | {os.cpu_count()} logical CPUs | {platform.system()} {platform.release()}"
    R.to_csv(os.path.join(OUT, "casper_t_runtime.csv"), index=False)
    print(R.groupby("Dataset", sort=False).agg(median_s=("Seconds", "median"), same=("same_as_main", "sum")).to_string())


if __name__ == "__main__":
    main(sys.argv[1:] or MAIN9)
