"""Worked example: selecting, inspecting, and validating with CASPER-t.

Run with::

    pip install casper-fs pandas
    python worked_example.py

It shows four things:
  1. which features CASPER-t keeps (by name),
  2. the per-feature reliability score t_f that drives the decision,
  3. dimensionality reduction via the standard scikit-learn selector API,
  4. leakage-free use inside a Pipeline + cross-validation, against the
     all-features baseline.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from casper import CasperT

warnings.filterwarnings("ignore")

# --- data (named columns) -------------------------------------------------
data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target
print(f"Data: {X.shape[0]} samples, {X.shape[1]} features\n")

# --- 1) standalone selection ---------------------------------------------
sel = CasperT(kappa=1.5, n_permutations=30, random_state=42).fit(X, y)
kept = X.columns[sel.get_support()]
print(f"Selected {sel.get_support().sum()}/{X.shape[1]} features:")
for f in kept:
    print("   -", f)

# --- 2) diagnostics: why these? the reliability score t_f -----------------
# Note: fit() converts the frame to an array, so map t_stats_ back to the
# original column names yourself to read them by feature.
print(f"\neps* (covering radius) = {sel.eps_star_} | base gamma = {sel.gamma_base_:.3f}")
tf = pd.Series(sel.t_stats_, index=X.columns).sort_values(ascending=False)
print("Top 5 features by t_f (kept when t_f > kappa):")
print(tf.head(5).round(2).to_string())

# --- 3) dimensionality reduction (standard selector API) ------------------
X_small = sel.transform(X)
print(f"\ntransform: {X.shape[1]} -> {X_small.shape[1]} columns")

# --- 4) leakage-free evaluation: refit per fold inside a Pipeline ---------
selected = Pipeline([
    ("casper", CasperT(random_state=42)),   # selection is refit on each train split
    ("scale", StandardScaler()),
    ("svm", SVC()),
])
baseline = Pipeline([("scale", StandardScaler()), ("svm", SVC())])

acc_sel = cross_val_score(selected, X, y, cv=5).mean()
acc_all = cross_val_score(baseline, X, y, cv=5).mean()

print("\n5-fold accuracy:")
print(f"   CASPER-t subset ({sel.get_support().sum()} features): {acc_sel * 100:.1f}%")
print(f"   all features   ({X.shape[1]} features): {acc_all * 100:.1f}%")
