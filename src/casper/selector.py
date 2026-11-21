"""CASPER-t: a scikit-learn compatible feature selector.

CASPER-t (Covering Approximation Space Permutation Elimination Rule, with a Monte Carlo
reliability margin t) discretizes each feature by unsupervised Jenks natural breaks, builds an
$L_1$ tolerance relation, selects an optimal tolerance threshold, scores every
feature by the drop in the $\\gamma$ dependency degree under $N$ permutations
(gamma-PFI), and retains a feature only when that drop exceeds the first
quartile of the feature-wise distribution by a reliable margin:

    t_f = (mean(Delta_gamma_f) - Q1) / SEM_f  >  kappa .

See the accompanying article for the full construction.  This estimator refits
everything on the training split it is given, so it is leakage-free inside a
scikit-learn ``Pipeline`` / cross-validation loop.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array

from ._core import (KAPPA, N_REP, MAX_N_TOPOLOGICAL, RANDOM_STATE,
                    tstat_support, stratified_subsample)
from ._fast import select_epsilon, gamma_pfi   # exact, bit-identical to _core
from .discretize import JenksDiscretizer, JENKS_GVF, JENKS_BINS


class CasperT(SelectorMixin, BaseEstimator):
    """Reliability-weighted topological feature selector.

    Parameters
    ----------
    kappa : float, default=1.5
        Reliability margin of the ``t_f`` rule.  Larger values retain fewer,
        more confidently-relevant features.
    n_permutations : int, default=30
        Number of permutation replicates per feature in gamma-PFI.
    max_bins : int, default=7
        Maximum number of Jenks bins per continuous feature.
    target_gvf : float, default=0.80
        Goodness-of-variance-fit target at which bin growth stops early.
    max_topological : int, default=3000
        Objects above which the O(n^2) topological stage is estimated on a
        class-stratified subsample (evaluation always uses the full data).
    random_state : int, default=42
        Seed for the Jenks k-means, the subsample, and the permutations.

    Attributes
    ----------
    support_ : ndarray of bool, shape (n_features,)
    delta_gamma_ : ndarray, shape (n_features,)   -- mean gamma drop per feature
    sem_ : ndarray, shape (n_features,)            -- its standard error
    t_stats_ : ndarray, shape (n_features,)        -- the t_f value per feature
    eps_star_ : float                              -- selected tolerance threshold
    gamma_base_ : float                            -- baseline gamma at eps_star_
    q1_ : float                                    -- first quartile of delta_gamma_
    """

    def __init__(self, kappa=KAPPA, n_permutations=N_REP, max_bins=JENKS_BINS,
                 target_gvf=JENKS_GVF, max_topological=MAX_N_TOPOLOGICAL,
                 random_state=RANDOM_STATE):
        self.kappa = kappa
        self.n_permutations = n_permutations
        self.max_bins = max_bins
        self.target_gvf = target_gvf
        self.max_topological = max_topological
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]

        # 1) Jenks discretization, fit on the training split only.
        self.discretizer_ = JenksDiscretizer(
            target_gvf=self.target_gvf, max_bins=self.max_bins,
            random_state=self.random_state)
        D = self.discretizer_.fit_transform(X)

        # 2) subsample only the O(n^2) topological stage.
        idx = stratified_subsample(y, self.max_topological, self.random_state)
        Dsub, ysub = D[idx], y[idx]

        # 3) optimal epsilon, then gamma-PFI, then the t_f rule.
        self.eps_star_, self.gamma_base_ = select_epsilon(Dsub, ysub)
        self.delta_gamma_, self.sem_ = gamma_pfi(
            Dsub, ysub, self.eps_star_, self.gamma_base_,
            n_rep=self.n_permutations, random_state=self.random_state)
        self.support_, self.q1_, self.t_stats_ = tstat_support(
            self.delta_gamma_, self.sem_, kappa=self.kappa)
        return self

    def _get_support_mask(self):
        check_is_fitted(self, "support_")
        return self.support_

    def _more_tags(self):
        return {"requires_y": True, "allow_nan": False}
