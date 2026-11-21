"""Quickstart: CASPER-t as a drop-in scikit-learn feature selector."""
from sklearn.datasets import load_breast_cancer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

from casper import CasperT

X, y = load_breast_cancer(return_X_y=True)

# --- standalone -----------------------------------------------------------
sel = CasperT(kappa=1.5, n_permutations=30, random_state=42).fit(X, y)
print(f"selected {sel.get_support().sum()} of {X.shape[1]} features")
print(f"eps* = {sel.eps_star_}, gamma_base = {sel.gamma_base_:.3f}")

# --- leakage-free inside a Pipeline + cross-validation --------------------
pipe = Pipeline([("casper", CasperT(random_state=42)), ("clf", SVC())])
scores = cross_val_score(pipe, X, y, cv=5)
print(f"5-fold accuracy with CASPER-t selection: {scores.mean():.3f}")
