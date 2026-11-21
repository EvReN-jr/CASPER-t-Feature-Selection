# Experiment core and tests

This folder holds the implementation used for the experiments of the article and the
tests cited in its appendix.

| File | Content |
|---|---|
| `casper_core.py` | distances, tolerance relation, minimal neighbourhoods, γ and γ^near, ε rule, γ-PFI, margin rule, `CasperSelector` |
| `casper_fast.py` | exact fast evaluation of γ on distinct code vectors (same values as the dense matrix product); `casper_core.pattern_gamma` extends it to γ^near, the δ-ball γ, every metric and the bootstrap |
| `exp_common.py` | data loading, fold-confined preprocessing, discretizers, outer folds, evaluation |
| `toy_example.py` | the worked example of the article (8 objects, 3 attributes) |
| `stats_utils.py` | corrected resampled t-test |
| `orig.py` | imports the reference pipeline `../pipeline/gamma_pfi_uhem.py` |
| `data/` | the five small raw datasets and the READY file used by `test_pipeline.py` |

Experiments of the article (each script writes per-fold CSV files under `../results/<experiment>/`):

| Script | Experiment |
|---|---|
| `run_main.py` | main comparison: CASPER-t candidates, classical, rough set and granular selectors, random-K, runtimes (`--lite` for the high-dimensional datasets) |
| `run_recent.py` | granular-ball selectors GBRS, GBFRS, GBNRS with the authors' code |
| `run_ablation.py` | component-wise ablation |
| `run_nperm.py` | number of shuffles N and bootstrap size B |
| `run_kappa_eps.py`, `run_kappa0.py` | threshold summary, κ sweep, κ = 0 |
| `run_kcurve.py` | accuracy against K for every ranking |
| `run_label_null.py`, `analyze_label_null.py` | label-permutation null of the γ drops |
| `run_pfi_rule.py` | margin rule applied to random-forest permutation importance |
| `run_subsample.py` | subsample size of the topological stage and scaling with n |
| `run_synthetic.py` | synthetic data with known relevant features, label noise |
| `run_mixed_noise.py` | mixed-type data with missing values, label noise |
| `fetch_raw.py` | downloads the five small raw datasets |
| `find_counterexamples.py` | small counterexamples for the propositions |
| `analyze_main.py`, `analyze_rest.py`, `make_tables.py` | summaries and the LaTeX tables (`../tables/`) |

`baselines.py` holds the competing selectors. GBRS, GBFRS and GBNRS are run with the
authors' published code, which is not redistributed here. Place it under `../third_party/`
(`GBRS/GBRS/Code`, `GBFRS_x/GBFRS_github`, `GBNRS_x/GBNRS_new`) or set `CASPER_THIRD_PARTY`.
The per-fold outputs behind every table of the article are in `../results/`.

Tests:

```bash
pip install numpy scipy pandas scikit-learn numba pytest
python -m pytest code/tests -q        # theory, worked example, pipeline
```

- `tests/test_theory.py` checks the propositions numerically (minimal neighbourhood,
  limits in ε, the Boolean matrix form against the definition, properties of γ-PFI).
- `tests/test_toy.py` recomputes every number of the worked example from the definitions.
- `tests/test_pattern_engine.py` checks that the exact engine equals the dense reference for every variant of the ablation.
- `tests/test_pipeline.py` checks that the fold pipeline is free of leakage and that
  `CasperSelector` reproduces ε*, Δγ, SE and the subset of the reference pipeline. It
  needs the packages of the reference pipeline (`xgboost`, `shap`, `skrebate`,
  `scikit-posthocs`, `openpyxl`, `jenkspy`) and is skipped without them.

The larger datasets are read from `data/<name>_READY_FOR_RS.xlsx` (set `CASPER_DATA` to
use another folder). `pipeline/gamma_pfi_uhem.py` prepares them from OpenML.
