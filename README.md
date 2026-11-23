# CASPER-t

[![PyPI](https://img.shields.io/pypi/v/casper-fs.svg?color=2a78d6)](https://pypi.org/project/casper-fs/)
[![Python](https://img.shields.io/pypi/pyversions/casper-fs.svg)](https://pypi.org/project/casper-fs/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![scikit-learn compatible](https://img.shields.io/badge/scikit--learn-compatible-eb6834.svg)](https://scikit-learn.org)

**C**overing **A**pproximation **S**pace **P**ermutation **E**limination **R**ule
(with a Monte Carlo reliability margin *t*) a **topological feature selector**,
packaged as a drop-in scikit-learn transformer.

> A model-free selector that fixes its own subset size. On 9 benchmark datasets
> CASPER-t retains **48–75 % of the features** (mean 60 %) at a **median accuracy
> loss of 0.93 points** against the full set. Random subsets of the same size lose
> 2.0 points, classical rankings 0.2–0.6 points.

```bash
pip install casper-fs
```

---

## How it works

CASPER-t treats each feature's usefulness as a *measured quantity with an error
bar*, and keeps a feature only when its measured contribution clears a reliability
margin not merely when it looks large once.

```mermaid
flowchart LR
    A["X, y"] --> B["Jenks<br/>discretization"]
    B --> C["L₁ tolerance<br/>relation"]
    C --> D["select ε*<br/>(covering radius)"]
    D --> E["γ-PFI<br/>N permutations"]
    E --> F["tᶠ = (Δγ − Q1) / SEM > κ"]
    F --> G["selected<br/>subset"]
```

1. **Discretize** each feature by unsupervised **Jenks natural breaks**.
2. Build an **L₁ tolerance relation** and pick the covering radius **ε\*** that
   maximises the γ dependency degree (just below saturation).
3. Score every feature by the **drop in γ under *N* permutations** (γ-PFI). A
   permutation-importance signal native to the covering approximation space.
4. Retain a feature only when that drop clears the first quartile of the
   feature-wise distribution **by a reliable margin**:

   > **tᶠ = (mean(Δγ\_f) − Q1) / SEM\_f > κ**   (default κ = 1.5)

Everything is refit on the training split it is given, so it is **leakage-free**
inside a scikit-learn `Pipeline` / cross-validation loop.

---

## Benchmark

**Protocol.** Training-fold-confined repeated cross-validation (5 folds × 2 repeats) on
**9 main datasets** (plus 7 datasets with 53–490 features and 4 mixed-type datasets),
evaluated with **7 classifiers**, against **18 competing selectors** (filter, wrapper,
embedded, rough-set and granular-ball families) and a random subset of the same size.
Baselines are compared at **matched cardinality**: each is truncated to CASPER-t's own
per-fold feature count, so every method is judged on the *same feature budget*.

| Dataset | Features kept (K / p) | CASPER-t acc. | All-features acc. | Δ |
|---|---|---|---|---|
| Cryotherapy   | 3.6 / 6   | 84.92 | 87.62 | −2.70 |
| Diabetes      | 6.6 / 10  | 76.85 | 75.94 | +0.91 |
| Breast Cancer | 12.5 / 26 | 93.61 | 96.17 | −2.56 |
| Parkinson's   | 9.4 / 19  | 85.64 | 85.31 | +0.33 |
| Sonar         | 31.4 / 60 | 78.30 | 79.57 | −1.27 |
| Banknote      | 3.0 / 4   | 96.55 | 96.82 | −0.28 |
| Phoneme       | 3.0 / 5   | 76.85 | 81.62 | −4.77 |
| HTRU2         | 4.3 / 7   | 96.44 | 96.89 | −0.44 |
| MAGIC         | 6.9 / 10  | 81.15 | 82.08 | −0.93 |

*Accuracy is the mean over the 7 classifiers and the 10 outer folds. K is the mean
per-fold subset size, p the median number of features after the redundancy filter.*

**On honest terms.** At a matched feature budget the rank tests do not separate
CASPER-t from the other selectors, and tree-based scores lead in mean rank
(CASPER-t: 12.5 of 19). Along the accuracy–K curve its ranking is significantly
better than a random ordering on Banknote and MAGIC and significantly worse on
Phoneme. A label-permutation null shows that the retained share is set by the
quartile cut rather than by the labels. With several hundred features the ranking
is no better than random, so CASPER-t is not recommended there. Per-fold results
are in [`results/`](results/) and [`detailed_tables/`](detailed_tables/).

---

## Quickstart

```python
from casper import CasperT
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)

# standalone
sel = CasperT(kappa=1.5, n_permutations=30, random_state=42).fit(X, y)
print(sel.get_support().sum(), "features selected")     # -> 13 of 30
print(sel.eps_star_, sel.t_stats_)                      # diagnostics

# leakage-free inside a Pipeline (refits per fold)
pipe = Pipeline([("casper", CasperT()), ("clf", SVC())])
cross_val_score(pipe, X, y, cv=5)
```

Requires Python ≥ 3.9, `numpy`, `scipy`, `scikit-learn`. Install from a clone with
`pip install .`. For the compiled kernel add the optional extra, `pip install "casper-fs[fast]"`
(installs `numba`).

### Speed

The dependency degree is evaluated by an exact fast engine (`casper/_fast.py`). It
gives the same ε\*, the same γ, the same Δγ and SEM for every replicate, and therefore
the same subset, as the direct implementation in `casper/_core.py`. The equality is bit for bit, and `tests/test_fast_equivalence.py`
checks it. Three facts remove work that cannot change the result:

1. Objects with the same code vector share their tolerance class, minimal
   neighbourhood and lower-approximation status. γ is computed on the distinct patterns,
   weighted by their counts. A pattern that holds two classes is never in the lower
   approximation.
2. The tolerance relation is reflexive and symmetric, so T(x) ⊆ T(z) forces z ∈ T(x).
   Only neighbours of another class need the inclusion test. With `numba`, packed bit
   rows run that test and stop at the first witness.
3. γ is a count divided by n, so it does not depend on the order of the objects.

Measured on the first outer fold, one γ-PFI fit (ε scan plus 30 permutations per
feature), 8 threads:

| Dataset | m | p | distinct rows | direct (s) | fast (s) |
|---|---|---|---|---|---|
| HTRU2 | 3000 | 8 | 137 | 14.3 | 0.17 |
| Phoneme | 3000 | 5 | 155 | 8.9 | 0.13 |
| MAGIC | 3000 | 10 | 1014 | 18.0 | 1.9 |
| Banknote | 1097 | 4 | 43 | 0.60 | 0.02 |
| Breast Cancer | 455 | 30 | 451 | 0.56 | 0.11 |

Under the measurement protocol of the article (one thread per fit, ten folds in parallel
on a 16-core workstation) a full fit on the nine main datasets takes 0.01–3.4 s. That is
faster than Boruta, RFE, ReliefF, standard permutation importance, GBNRS and GBRS on every
dataset, and on most datasets slower than plain filter scores (ANOVA, mutual information,
LASSO, HSIC-Lasso) and QuickReduct. `code/run_runtime.py` times one fit per fold with one
thread on your own machine (on a four-core laptop: 0.12–7.5 s).

Without `numba` the NumPy fallback is used. It is also exact, only slower.

## API

`CasperT(kappa=1.5, n_permutations=30, max_bins=7, target_gvf=0.80, max_topological=3000, random_state=42)`

| Parameter | Meaning |
|---|---|
| `kappa` | reliability margin of the tᶠ rule; larger ⇒ fewer, more confident features |
| `n_permutations` | permutation replicates per feature in γ-PFI |
| `max_bins` / `target_gvf` | Jenks bin cap / goodness-of-variance-fit early-stop target |
| `max_topological` | subsample cap for the topological stage (evaluation uses full data) |

Fitted attributes: `support_`, `delta_gamma_`, `sem_`, `t_stats_`, `eps_star_`,
`gamma_base_`, `q1_`. Standard `SelectorMixin` API: `fit`, `transform`,
`get_support`, `get_feature_names_out`.

## Reproducing the article

`pipeline/gamma_pfi_uhem.py` is the full cross-validation pipeline used for the paper
(all baselines, statistics, and per-fold outputs); `pipeline/make_tables.py`
regenerates the LaTeX tables from those outputs.
`code/` holds the experiment core (`CasperSelector`, the exact fast evaluation) and the
tests cited in the article's appendix, see `code/README.md`.

```bash
pip install -r requirements.txt          # versions of the verification environment
python -m pytest tests code/tests -q     # on Windows set PYTHONUTF8=1 first
cd code && python run_runtime.py         # times every fold and checks it against results/main
```

With the versions in `requirements.txt`, `run_runtime.py` reproduces the stored
selections of all ten folds on eight of the nine main datasets and on six of the ten
folds of Diabetes, where k-means returns slightly different Jenks centres. The analyses
added in the revision are `run_label_null.py` (L = 100, all folds),
`run_labelnull_overlap.py`, `run_synthetic_eps.py` and `run_reference_percentile.py`.

### Detailed results (`detailed_tables/`)

The article's tables report means only, for readability. These files hold the same
numbers **with their dispersion and at fold-level granularity**:

| File | What it is |
|---|---|
| `summary_mean_std.csv` | one row per (dataset × method × metric): mean, fold std, across-classifier std, retained `K`, `p`, retained % |
| `fold_level_hybrid.csv` | every outer fold, HYBRID classifier only |
| `fold_level_raw.csv.gz` | every outer fold × 8 classifiers × every method |

Method ids: `V1_tstat` = CASPER-t; `*_matchK` = baselines truncated at CASPER-t's
per-fold cardinality; `*_native` = rough-set baselines at their own stopping point;
`ALL` = full feature set.

## Citation

If you use CASPER-t, please cite the article:

```bibtex
@article{casper_t,
  title  = {A Topological Feature Selection Framework via Generalized Covering
            Approximation Spaces with Permutation Importance},
  author  = {Yiğit, Uğur and Boyabatlı, Kenan Evren and Çuhadar, Ebrar and
             Elgün, Esma Nur and Sönmez, Ertan},
  journal = {Preprint},
  year    = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
