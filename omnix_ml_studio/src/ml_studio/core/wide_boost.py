"""CatBoost with settings that keep it fast on wide (omics) data.

With hundreds to thousands of features CatBoost's defaults (254 split borders, every feature
considered at every split) make it by far the slowest model: ~66 s per fold on 1,000 genes and
over 15 minutes for a 5-fold run on 10,962 genes. Above ``WIDE`` features, and only for settings
the user has not set, it uses 32 borders and evaluates a random ~100 features per split level
(``rsm``), which was 50× faster with the same cross-validated AUC on a TCGA tumour/normal set.
"""
from __future__ import annotations

WIDE = 500


def _adapt(est, X):
    p = X.shape[1]
    if p <= WIDE:
        return
    prm = est.get_params()
    upd = {}
    if prm.get("border_count") is None:
        upd["border_count"] = 32
    if prm.get("rsm") is None:
        upd["rsm"] = float(min(1.0, max(0.02, 100.0 / p)))
    if upd:
        est.set_params(**upd)


try:
    from catboost import CatBoostClassifier, CatBoostRegressor

    class WideCatBoostClassifier(CatBoostClassifier):
        def fit(self, X, y=None, **kw):
            _adapt(self, X)
            return super().fit(X, y, **kw)

    class WideCatBoostRegressor(CatBoostRegressor):
        def fit(self, X, y=None, **kw):
            _adapt(self, X)
            return super().fit(X, y, **kw)
except ImportError:      # optional dependency
    WideCatBoostClassifier = WideCatBoostRegressor = None
