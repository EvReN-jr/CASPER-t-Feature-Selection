# -*- coding: utf-8 -*-
"""
Son yılların granular-ball kaba küme seçicileri, yazar kodlarıyla:
  GBRS (Xia vd., IEEE TNNLS 2023/2025), GBFRS (Lian, Xia vd., IEEE TNNLS 2026), GBNRS-new (Xia vd.; TKDE 2022 kod tabanı).
Her dış katta: aynı kat-içi ön işleme; CASPER-t K'sı (eşlenik karşılaştırma için); her yöntem kendi K'sında ve CASPER-t K'sında.
Kat başına yöntem süre sınırı (varsayılan 1800 s); aşılırsa 'timeout' kaydedilir (SelectArena protokolü gibi).
Çıktı formatı run_main ile aynı → analyze_main ana tablolara birleştirir.
"""
import json, time, signal, argparse
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from exp_common import load, folds, FoldPreprocessor, Discretizer, evaluate, eval_rows, set_threads, save_rows, MAIN9
from casper_core import CasperSelector
import baselines as B


class _TO(Exception):
    pass


def _alarm(signum, frame):
    raise _TO()


def run_fold(ds, fi, tr, te, X, y, budget):
    set_threads(1)
    signal.signal(signal.SIGALRM, _alarm)
    pp = FoldPreprocessor().fit(X.iloc[tr])
    Xtr = pp.transform(X.iloc[tr]).reset_index(drop=True); Xte = pp.transform(X.iloc[te]).reset_index(drop=True)
    ytr, yte = y[tr], y[te]; cols = list(Xtr.columns); p = len(cols)
    Xd = Discretizer().fit(Xtr).transform(Xtr).values.astype(float)
    K = len(CasperSelector().fit(Xd, ytr).support_idx_)
    rows, sel = [], []
    for f in (B.gbnrs, B.gbfrs, B.gbrs):
        signal.alarm(budget)
        try:
            r, o, s = f(Xtr, ytr)
            signal.alarm(0)
        except _TO:
            sel.append(dict(Dataset=ds, Fold=fi, Method=f.__name__, K=np.nan, p=p, Seconds=float(budget), status="timeout", Features="[]"))
            print(f"[{ds} f{fi}] {f.__name__} TIMEOUT", flush=True); continue
        except Exception as e:
            signal.alarm(0)
            sel.append(dict(Dataset=ds, Fold=fi, Method=f.__name__, K=np.nan, p=p, Seconds=np.nan, status=f"error:{e}", Features="[]"))
            continue
        for kind, idx in (("matched", r[:K]), ("own", o)):
            res, k = evaluate(Xtr, Xte, ytr, yte, [cols[i] for i in idx])
            rows.extend(eval_rows(dict(Dataset=ds, Fold=fi, Method=f.__name__, Kind=kind), res, k))
            sel.append(dict(Dataset=ds, Fold=fi, Method=f.__name__ + ("[own]" if kind == "own" else ""), K=len(idx), p=p,
                            Seconds=s, status="ok", kind=kind, Features=json.dumps([cols[i] for i in idx])))
    print(f"[{ds}] fold {fi} bitti", flush=True)
    return rows, sel


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("datasets", nargs="*", default=MAIN9)
    ap.add_argument("--jobs", type=int, default=5); ap.add_argument("--budget", type=int, default=1800); a = ap.parse_args()
    tasks = []
    data = {ds: load(ds) for ds in a.datasets}
    for ds in a.datasets:
        X, y = data[ds]
        tasks += [(ds, i + 1, tr, te) for i, (tr, te) in enumerate(folds(X, y))]
    out = Parallel(n_jobs=a.jobs)(delayed(run_fold)(ds, fi, tr, te, *data[ds], a.budget) for ds, fi, tr, te in tasks)
    for ds in a.datasets:
        rows = [r for o in out for r in o[0] if r["Dataset"] == ds]; sel = [s for o in out for s in o[1] if s["Dataset"] == ds]
        save_rows(rows, "recent", f"{ds}_metrics"); save_rows(sel, "recent", f"{ds}_selection")
    print("=== recent tamam", flush=True)


if __name__ == "__main__":
    main()
