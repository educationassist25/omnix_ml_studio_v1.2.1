"""Feature selection inside training folds, stability analysis, and univariate statistics.

* :class:`FeatureSelector` is a scikit-learn selector that is placed in the modelling
  pipeline, so it is refit on every training fold (never on validation / holdout rows).
* :func:`selection_stability` repeats preprocessing + selection on resampled training
  sets and reports how often each feature is chosen (frequency), its mean rank, and
  stability indices (pairwise Jaccard, Kuncheva).
* :func:`univariate_statistics` gives *statistical association* (tests + BH-FDR) on the
  development rows. This is reported separately: a feature can be predictive in a
  multivariable model without being individually significant, and vice versa.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, replace
from typing import Callable, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier,
                              RandomForestRegressor)
from sklearn.feature_selection import (RFE, RFECV, SelectorMixin, f_classif, f_regression,
                                       mutual_info_classif, mutual_info_regression)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, LogisticRegression, Ridge, lasso_path
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.svm import l1_min_c

from .config import PreprocessConfig, ValidationConfig
from .preprocessing import PreparedData, _prep_steps
from .splitting import make_cv

METHODS = {
    "variance": "Variance (unsupervised top-k)",
    "univariate": "Univariate F-test",
    "mutual_info": "Mutual information",
    "l1": "L1 / lasso path (entry order)",
    "model": "Model-based (|coef|, L2 linear)",
    "tree": "Tree-based (Extra Trees importance)",
    "rfe": "Recursive feature elimination (RFE)",
    "rfecv": "RFECV (number chosen by inner CV)",
    "permutation": "Permutation importance (inner split)",
}
PERMUTATION_MAX_FEATURES = 300


def _z(X):
    sd = X.std(0)
    return (X - X.mean(0)) / np.where(sd == 0, 1, sd)


class FeatureSelector(SelectorMixin, BaseEstimator):
    """Select ``k`` features by ``method``. Higher ``scores_`` = more important.

    Fitted only on the data passed to ``fit`` — inside a pipeline that is the
    training fold. Linear-model based methods standardise internally for scoring.
    """

    def __init__(self, method: str = "univariate", k: int = 20, task: str = "binary",
                 random_state: int = 0, step: float = 0.1, inner_cv: int = 3):
        self.method, self.k, self.task = method, k, task
        self.random_state, self.step, self.inner_cv = random_state, step, inner_cv

    # -- helpers
    def _linear(self, l1=False, C=1.0):
        if self.task == "regression":
            return Lasso(alpha=0.05, max_iter=20000) if l1 else Ridge(alpha=1.0)
        return LogisticRegression(penalty="l1" if l1 else "l2", C=C,
                                  solver="liblinear" if (l1 and self.task == "binary") else
                                  ("saga" if l1 else "lbfgs"), max_iter=5000,
                                  random_state=self.random_state)

    def fit(self, X, y):
        X, y = self._validate_data(X, y, reset=True, force_all_finite=True)
        y = np.asarray(y)
        p = X.shape[1]
        k = int(min(self.k, p))
        m = self.method
        clf = self.task != "regression"
        rs = self.random_state
        if m == "variance":
            s = X.var(0)
        elif m == "univariate":
            s = (f_classif if clf else f_regression)(X, y)[0]
        elif m == "mutual_info":
            s = (mutual_info_classif if clf else mutual_info_regression)(X, y, random_state=rs)
        elif m == "model":
            est = self._linear().fit(_z(X), y)
            s = np.abs(np.atleast_2d(est.coef_)).max(0)
        elif m == "l1":
            s = self._l1_entry_scores(_z(X), y, k)
        elif m == "tree":
            T = ExtraTreesClassifier if clf else ExtraTreesRegressor
            s = T(n_estimators=250, random_state=rs, n_jobs=1).fit(X, y).feature_importances_
        elif m == "rfe":
            r = RFE(self._linear(), n_features_to_select=k, step=self.step).fit(_z(X), y)
            s = -r.ranking_.astype(float)
        elif m == "rfecv":
            cv = (StratifiedKFold if clf else KFold)(self.inner_cv, shuffle=True, random_state=rs)
            r = RFECV(self._linear(), step=self.step, cv=cv, min_features_to_select=1,
                      scoring="roc_auc" if self.task == "binary" else
                      ("balanced_accuracy" if clf else "r2"), n_jobs=1).fit(_z(X), y)
            s = -r.ranking_.astype(float)
            k = int(r.n_features_)
        elif m == "permutation":
            if p > PERMUTATION_MAX_FEATURES:
                raise ValueError(f"Permutation selection is limited to {PERMUTATION_MAX_FEATURES} "
                                 f"features (got {p}); add a univariate pre-filter first.")
            Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.3, random_state=rs,
                                              stratify=y if clf else None)
            M = RandomForestClassifier if clf else RandomForestRegressor
            est = M(n_estimators=200, random_state=rs, n_jobs=1).fit(Xa, ya)
            s = permutation_importance(est, Xb, yb, n_repeats=5, random_state=rs, n_jobs=1).importances_mean
        else:
            raise ValueError(f"Unknown selection method '{m}'. Options: {list(METHODS)}")
        s = np.nan_to_num(np.asarray(s, float), nan=-np.inf)
        order = np.lexsort((np.arange(p), -s))          # stable: ties keep column order
        self.scores_ = s
        self.ranking_ = np.empty(p, int); self.ranking_[order] = np.arange(1, p + 1)
        self.k_ = k
        mask = np.zeros(p, bool); mask[order[:k]] = True
        self.support_ = mask
        return self

    def _l1_entry_scores(self, X, y, k):
        """Score = order of entry along the L1 regularisation path (earlier = better)."""
        p = X.shape[1]
        entry = np.full(p, np.inf)
        if self.task == "regression":
            alphas, coefs, _ = lasso_path(X, y - y.mean(), n_alphas=60, eps=1e-3)
            for j in range(len(alphas)):              # alphas descending
                nz = np.abs(coefs[:, j]) > 0
                entry[nz & np.isinf(entry)] = j
            last = np.abs(coefs[:, -1])
        else:
            cmin = l1_min_c(X, y, loss="log")
            last = np.zeros(p)
            for j, C in enumerate(cmin * np.logspace(0, 3, 30)):
                est = self._linear(l1=True, C=C).fit(X, y)
                c = np.abs(np.atleast_2d(est.coef_)).max(0)
                entry[(c > 0) & np.isinf(entry)] = j
                last = c
                if (~np.isinf(entry)).sum() >= k:
                    break
        # earlier entry -> higher score; tie-break by final |coef|
        e = np.where(np.isinf(entry), 1e6, entry)
        return -e + last / (last.max() + 1e-12) * 0.5

    def _get_support_mask(self):
        return self.support_

    def _more_tags(self):
        return {"allow_nan": False}


# --------------------------------------------------------------------------- utilities
def to_original(name: str, categorical: list[str]) -> str:
    """Map an encoded column (e.g. 'stage_III') back to its original feature ('stage')."""
    for c in sorted(categorical, key=len, reverse=True):
        if name == c or name.startswith(c + "_"):
            return c
    return name


def selected_features(pipe) -> Optional[list[str]]:
    """Encoded names of the features the fitted pipeline's model actually receives."""
    steps = dict(pipe.steps)
    names = np.asarray(steps["columns"].get_feature_names_out())
    for key in ("select", "fs"):
        if key in steps:
            names = names[steps[key].get_support()]
    return list(names)


# --------------------------------------------------------------------------- stability
@dataclass
class StabilityResult:
    frequency: pd.DataFrame        # rows = feature (encoded), columns = method -> fraction of resamples
    mean_rank: pd.DataFrame        # rows = feature, columns = method
    indices: pd.DataFrame          # per method: n_resamples, k, mean Jaccard, Kuncheva
    overlap: pd.DataFrame          # method x method Jaccard of consensus top-k sets
    selections: dict               # method -> list of selected-feature lists
    n_features: int
    settings: dict


def kuncheva(sets: list[set], k: int, p: int) -> float:
    """Kuncheva consistency index (requires equal-size subsets). 0 ≈ chance, 1 = identical."""
    if k <= 0 or k >= p or len(sets) < 2:
        return np.nan
    vals = []
    for a, b in itertools.combinations(sets, 2):
        r = len(a & b)
        vals.append((r * p - k * k) / (k * (p - k)))
    return float(np.mean(vals))


def mean_jaccard(sets: list[set]) -> float:
    if len(sets) < 2:
        return np.nan
    return float(np.mean([len(a & b) / max(len(a | b), 1) for a, b in itertools.combinations(sets, 2)]))


_REPEATED = {"stratified_kfold": "repeated_stratified_kfold", "kfold": "repeated_kfold"}


def selection_stability(data: PreparedData, idx: np.ndarray, task: str, methods: list[str], k: int,
                        prep_cfg: PreprocessConfig, val_cfg: ValidationConfig, n_repeats: int = 3,
                        progress: Optional[Callable[[int, int, str], None]] = None) -> StabilityResult:
    """Run preprocessing + each selector on every CV training set (repeated CV)."""
    idx = np.asarray(idx)
    X, y = data.X.iloc[idx], data.y[idx]
    groups = data.groups[idx] if data.groups is not None else None
    cfg = replace(val_cfg, cv_strategy=_REPEATED.get(val_cfg.cv_strategy, val_cfg.cv_strategy),
                  n_repeats=max(1, n_repeats))
    cv = make_cv(cfg, task, groups is not None)
    splits = list(cv.split(np.zeros(len(idx)), y, groups))
    # preprocessing without the selection step under study
    base_cfg = replace(prep_cfg, fs_method=None)
    sel = {m: [] for m in methods}
    ranks = {m: [] for m in methods}
    names_all = None
    for i, (tr, _te) in enumerate(splits):
        if progress:
            progress(i, len(splits), f"resample {i + 1}/{len(splits)}")
        steps = _prep_steps(data.numeric, data.categorical, base_cfg, task)
        Z = X.iloc[tr]
        for _, stp in steps:                       # fitted on this training set only
            Z = clone(stp).fit_transform(Z, y[tr])
        Z = pd.DataFrame(Z)
        names = list(Z.columns)
        names_all = names if names_all is None else names_all
        for m in methods:
            fs = FeatureSelector(m, k, task, random_state=val_cfg.random_state + i).fit(Z, y[tr])
            sel[m].append([names[j] for j in np.where(fs.support_)[0]])
            ranks[m].append(pd.Series(fs.ranking_, index=names))
    n = len(splits)
    all_feats = sorted({f for m in methods for s in sel[m] for f in s} | set(names_all or []))
    freq = pd.DataFrame(0.0, index=all_feats, columns=methods)
    mrank = pd.DataFrame(np.nan, index=all_feats, columns=methods)
    rows = []
    for m in methods:
        for s in sel[m]:
            freq.loc[s, m] += 1.0 / n
        mrank[m] = pd.concat(ranks[m], axis=1).mean(axis=1).reindex(all_feats)
        sets = [set(s) for s in sel[m]]
        kk = int(np.median([len(s) for s in sets]))
        p = len(names_all or [])
        rows.append({"method": m, "label": METHODS[m], "n_resamples": n, "k (median)": kk,
                     "mean_jaccard": mean_jaccard(sets),
                     "kuncheva": kuncheva(sets, kk, p) if len({len(s) for s in sets}) == 1 else np.nan,
                     "n_features_ever_selected": int((freq[m] > 0).sum())})
    cons = {m: set(freq[m].sort_values(ascending=False).index[:k]) for m in methods}
    ov = pd.DataFrame([[len(cons[a] & cons[b]) / max(len(cons[a] | cons[b]), 1) for b in methods]
                       for a in methods], index=methods, columns=methods)
    order = freq.mean(axis=1).sort_values(ascending=False).index
    return StabilityResult(freq.loc[order], mrank.loc[order], pd.DataFrame(rows), ov, sel,
                           len(names_all or []), {"k": k, "n_resamples": n, "cv": type(cv).__name__,
                                                  "methods": methods})


# --------------------------------------------------------------------------- statistics
def univariate_statistics(data: PreparedData, idx: np.ndarray, task: str) -> pd.DataFrame:
    """Per-feature association with the outcome on development rows, with BH-FDR.

    Binary: Mann–Whitney U (effect = Cliff's delta). Multiclass: Kruskal–Wallis
    (effect = epsilon²). Regression: Spearman ρ. Categorical features: χ² (effect =
    Cramér's V) for classification, Kruskal–Wallis for regression.
    """
    from scipy import stats
    from statsmodels.stats.multitest import multipletests
    idx = np.asarray(idx)
    X, y = data.X.iloc[idx], data.y[idx]
    rows = []
    for c in data.numeric:
        x = X[c].to_numpy(float)
        ok = np.isfinite(x)
        xv, yv = x[ok], y[ok]
        try:
            if task == "binary":
                a, b = xv[yv == 1], xv[yv == 0]
                u, pval = stats.mannwhitneyu(a, b, alternative="two-sided")
                eff, test = 2 * u / (len(a) * len(b)) - 1, "Mann–Whitney U (Cliff's δ)"
            elif task == "multiclass":
                gs = [xv[yv == g] for g in np.unique(yv)]
                h, pval = stats.kruskal(*gs)
                eff, test = (h - len(gs) + 1) / (len(xv) - len(gs)), "Kruskal–Wallis (ε²)"
            else:
                eff, pval = stats.spearmanr(xv, yv)
                test = "Spearman ρ"
        except ValueError:
            eff, pval, test = np.nan, 1.0, "not testable (constant)"
        rows.append({"feature": c, "type": "numeric", "test": test, "effect": eff, "p": pval,
                     "n": int(ok.sum())})
    for c in data.categorical:
        s = X[c].astype(str)
        if task == "regression":
            gs = [y[s.values == g] for g in s.unique()]
            try:
                h, pval = stats.kruskal(*gs) if len(gs) > 1 else (np.nan, 1.0)
            except ValueError:
                h, pval = np.nan, 1.0
            rows.append({"feature": c, "type": "categorical", "test": "Kruskal–Wallis", "effect": h,
                         "p": pval, "n": len(s)})
        else:
            ct = pd.crosstab(s, y)
            if ct.shape[0] < 2:
                rows.append({"feature": c, "type": "categorical", "test": "not testable",
                             "effect": np.nan, "p": 1.0, "n": len(s)}); continue
            chi2, pval = stats.chi2_contingency(ct, correction=False)[:2]
            v = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
            rows.append({"feature": c, "type": "categorical", "test": "χ² (Cramér's V)", "effect": v,
                         "p": pval, "n": len(s)})
    out = pd.DataFrame(rows)
    out["p"] = out["p"].fillna(1.0)
    out["q_bh"] = multipletests(out["p"], method="fdr_bh")[1]
    return out.sort_values("p").reset_index(drop=True)
