"""Unsupervised Jenks natural-breaks discretization (k-means, one dimension).

Bins are grown until the Goodness-of-Variance-Fit reaches ``target_gvf`` or a
maximum of ``max_bins``.  ``fit`` is called on the training split only; the
learned bin centres are then applied to train and test alike, so the
discretization introduces no label leakage.  Binary / low-cardinality columns
are passed through as ordinal codes.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

JENKS_GVF = 0.80
JENKS_BINS = 7


def _is_categorical(col):
    col = np.asarray(col)
    if col.dtype == object or col.dtype == bool:
        return True
    if np.issubdtype(col.dtype, np.integer) and len(np.unique(col)) <= 2:
        return True
    return len(np.unique(col)) <= 2


def _gvf(x, labels, centers):
    mean = x.mean()
    sdam = ((x - mean) ** 2).sum()
    if sdam == 0:
        return 1.0
    sdcm = sum(((x[labels == i] - c) ** 2).sum() for i, c in enumerate(centers))
    return (sdam - sdcm) / sdam


class JenksDiscretizer:
    """Column-wise k-means (Jenks) discretization to ordered interval codes."""

    def __init__(self, target_gvf=JENKS_GVF, max_bins=JENKS_BINS, random_state=42):
        self.target_gvf = target_gvf
        self.max_bins = max_bins
        self.random_state = random_state

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.n_features_in_ = X.shape[1]
        self.centers_ = {}          # col idx -> sorted bin centres (continuous cols)
        for j in range(X.shape[1]):
            col = X[:, j]
            uniq = np.unique(col)
            if _is_categorical(col) or len(uniq) <= 2:
                continue            # pass-through as ordinal codes
            best_centers = uniq[:1]
            for k in range(2, min(self.max_bins + 1, len(uniq))):
                km = KMeans(n_clusters=k, random_state=self.random_state, n_init=5)
                lbls = km.fit_predict(col.reshape(-1, 1))
                ctrs = km.cluster_centers_.flatten()
                best_centers = ctrs
                if _gvf(col, lbls, ctrs) >= self.target_gvf:
                    break
            self.centers_[j] = np.sort(best_centers)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        out = X.copy()
        for j in range(X.shape[1]):
            if j in self.centers_:
                c = self.centers_[j]
                out[:, j] = np.abs(X[:, [j]] - c.reshape(1, -1)).argmin(axis=1) + 1
        return out

    def fit_transform(self, X):
        return self.fit(X).transform(X)
