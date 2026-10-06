"""Holdout evaluation, bootstrap confidence intervals, nested CV, external validation.

Design rules enforced here:

* The holdout is evaluated only with an already-fitted pipeline; nothing is fitted on it.
  Each evaluation is logged in a :class:`HoldoutLedger` so repeated peeking is visible.
* Nested CV: the outer test fold is never passed to model selection, tuning, feature
  selection or preprocessing — those happen inside the inner loop on outer-train rows.
* External validation uses the saved pipeline as-is (no refit) and reports feature
  compatibility and dataset shift separately from performance.
* Bootstrap CIs resample the independent unit (groups when present).
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd
from sklearn.base import clone

from . import metrics as MET
from .config import PreprocessConfig, StudyDesign, ValidationConfig
from .runner import resolve_jobs
from .model_comparison import registry
from .optimization import inner_config, tune
from .preprocessing import PreparedData, build_pipeline, prepare_xy
from .splitting import make_cv


# --------------------------------------------------------------------------- predictions
def predict_scores(pipe, X: pd.DataFrame, task: str):
    """(y_pred, proba or None).

    For binary models without probabilities, ``proba`` holds sigmoid(decision function):
    monotone scores valid for ROC/PR analysis but not calibrated (see ``has_proba``)."""
    if task == "regression":
        return pipe.predict(X), None
    if MET.has_proba(pipe):
        proba = pipe.predict_proba(X)
        return np.argmax(proba, 1), proba
    y_pred = pipe.predict(X)
    if task == "binary" and hasattr(pipe, "decision_function"):
        s = 1 / (1 + np.exp(-np.clip(np.asarray(pipe.decision_function(X), float), -30, 30)))
        return y_pred, np.c_[1 - s, s]
    return y_pred, None


# --------------------------------------------------------------------------- bootstrap
def bootstrap_ci(task: str, y, y_pred, proba, n_classes: int = 2, groups=None,
                 n_boot: int = 1000, alpha: float = 0.05, random_state: int = 0,
                 calibrated: bool = True) -> pd.DataFrame:
    """Percentile bootstrap CIs; resamples groups (clusters) when given."""
    y = np.asarray(y); y_pred = np.asarray(y_pred)
    point = MET.compute(task, y, y_pred, proba, n_classes, calibrated)
    rng = np.random.default_rng(random_state)
    units = np.unique(groups) if groups is not None else None
    members = {g: np.where(groups == g)[0] for g in units} if units is not None else None
    draws = {k: [] for k in point}
    for _ in range(n_boot):
        if units is not None:
            pick = rng.choice(units, len(units), replace=True)
            ii = np.concatenate([members[g] for g in pick])
        else:
            ii = rng.integers(0, len(y), len(y))
        if task != "regression" and len(np.unique(y[ii])) < 2:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = MET.compute(task, y[ii], y_pred[ii], None if proba is None else proba[ii], n_classes,
                            calibrated)
        for k in point:
            if k in m:
                draws[k].append(m[k])
    rows = []
    for k, v in point.items():
        d = np.asarray(draws[k])
        lo, hi = (np.quantile(d, [alpha / 2, 1 - alpha / 2]) if len(d) >= 50 else (np.nan, np.nan))
        rows.append({"metric": k, "label": MET.METRIC_LABELS.get(k, k), "estimate": v,
                     "ci_low": lo, "ci_high": hi, "n_boot": len(d)})
    out = pd.DataFrame(rows)
    out.attrs["unit"] = "group" if units is not None else "sample"
    out.attrs["n_units"] = len(units) if units is not None else len(y)
    return out


# --------------------------------------------------------------------------- final model
def fit_final(data: PreparedData, idx: np.ndarray, task: str, model_key: str,
              prep_cfg: PreprocessConfig, val_cfg: ValidationConfig, params: Optional[dict] = None):
    """Fit preprocessing + model on the given (development) rows only."""
    spec = registry(task).get(model_key)
    pipe = build_pipeline(spec.factory(val_cfg.random_state), data.numeric, data.categorical,
                          prep_cfg, task, val_cfg.random_state)
    if params:
        pipe.set_params(**params)
    idx = np.asarray(idx)
    pipe.fit(data.X.iloc[idx], data.y[idx])
    return pipe


@dataclass
class HoldoutLedger:
    """Records every evaluation of the sealed holdout set."""
    entries: list = field(default_factory=list)

    def record(self, model_desc: str, primary: float):
        self.entries.append({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "model": model_desc,
                             "primary_metric": primary})

    @property
    def n_evaluations(self) -> int:
        return len(self.entries)

    def warning(self) -> Optional[str]:
        if self.n_evaluations > 1:
            return (f"The holdout has been evaluated {self.n_evaluations} times. Choosing a model "
                    "because it scored best on the holdout turns the holdout into a validation "
                    "set, and its estimate becomes optimistic. Report the first evaluation of "
                    "the pre-specified model.")
        return None


def evaluate_holdout(pipe, data: PreparedData, test_idx: np.ndarray, task: str,
                     n_boot: int = 1000, random_state: int = 0,
                     ledger: Optional[HoldoutLedger] = None, model_desc: str = "") -> dict:
    test_idx = np.asarray(test_idx)
    if len(test_idx) == 0:
        raise ValueError("No holdout set was reserved.")
    X, y = data.X.iloc[test_idx], data.y[test_idx]
    g = data.groups[test_idx] if data.groups is not None else None
    y_pred, proba = predict_scores(pipe, X, task)
    nc = len(data.classes) if data.classes else 2
    ci = bootstrap_ci(task, y, y_pred, proba, nc, g, n_boot, random_state=random_state,
                      calibrated=MET.has_proba(pipe))
    pm = MET.primary_metric(task)
    if ledger is not None:
        ledger.record(model_desc, float(ci.set_index("metric").loc[pm, "estimate"])
                      if pm in set(ci["metric"]) else np.nan)
    return {"metrics": ci, "y_true": y, "y_pred": y_pred, "proba": proba, "groups": g,
            "n": len(y), "warnings": _sample_size_warnings(task, y, g)}


def _sample_size_warnings(task, y, groups) -> list[str]:
    w = []
    n_units = len(np.unique(groups)) if groups is not None else len(y)
    if n_units < 50:
        w.append(f"Only {n_units} independent units in this evaluation set; confidence "
                 "intervals are wide and the point estimate is unstable.")
    if task != "regression":
        cnt = np.bincount(np.asarray(y).astype(int))
        if cnt[cnt > 0].min() < 10:
            w.append(f"The smallest class has {cnt[cnt > 0].min()} samples in this set.")
    return w


# --------------------------------------------------------------------------- nested CV
@dataclass
class NestedCVResult:
    outer: pd.DataFrame                 # one row per outer fold
    summary: pd.DataFrame               # metric mean / sd across outer folds
    selection_frequency: pd.Series      # how often each model won the inner loop
    selected_features: list             # per outer fold (when univariate selection used)
    settings: dict
    outer_splits: list


def nested_cv(data: PreparedData, idx: np.ndarray, task: str, model_keys: list[str],
              prep_cfg: PreprocessConfig, val_cfg: ValidationConfig, n_iter: int = 10,
              progress: Optional[Callable[[int, int, str], None]] = None) -> NestedCVResult:
    """Outer CV estimates the performance of the *whole* selection procedure
    (choose model + tune hyperparameters + select features) on unseen folds."""
    idx = np.asarray(idx)
    X, y = data.X.iloc[idx], data.y[idx]
    groups = data.groups[idx] if data.groups is not None else None
    outer_cv = make_cv(inner_config(val_cfg), task, groups is not None)
    outer = list(outer_cv.split(np.zeros(len(idx)), y, groups))
    nc = len(data.classes) if data.classes else 2
    pm = MET.primary_metric(task)
    reg = registry(task)
    sel_metric = pm
    if task == "multiclass" and not all(reg.get(k).proba for k in model_keys):
        sel_metric = "balanced_accuracy"      # same selection metric for every candidate
    rows, feats = [], []
    sub = PreparedData(X, y, data.classes, groups, data.numeric, data.categorical, data.index[idx])
    for k, (tr, te) in enumerate(outer):
        if progress:
            progress(k, len(outer), f"outer fold {k + 1}/{len(outer)}")
        best = None
        inner_scores = {}
        for key in model_keys:
            try:
                r = tune(sub, tr, task, key, prep_cfg, val_cfg, n_iter=n_iter, metric=sel_metric,
                         n_folds=val_cfg.inner_folds, random_state=val_cfg.random_state + k)
            except Exception as e:  # noqa: BLE001
                inner_scores[key] = f"failed: {e}"
                continue
            inner_scores[key] = r.best_score
            if best is None or (r.best_score > best.best_score):   # scorers are 'greater is better'
                best = r
        if best is None:
            raise RuntimeError(f"All models failed in outer fold {k + 1}: {inner_scores}")
        y_pred, proba = predict_scores(best.estimator, X.iloc[te], task)
        m = MET.compute(task, y[te], y_pred, proba, nc, MET.has_proba(best.estimator))
        if "select" in best.estimator.named_steps or "fs" in best.estimator.named_steps:
            from .feature_selection import selected_features
            feats.append(selected_features(best.estimator))
        rows.append({"outer_fold": k + 1, "n_train": len(tr), "n_test": len(te),
                     "selected_model": best.model_key,
                     "inner_score": best.best_score,
                     "params": {kk.replace("model__", ""): (round(v, 5) if isinstance(v, float) else v)
                                for kk, v in best.best_params.items()},
                     **{f"outer_{kk}": v for kk, v in m.items()},
                     "inner_scores": inner_scores})
    df = pd.DataFrame(rows)
    mcols = [c for c in df.columns if c.startswith("outer_") and c != "outer_fold"]
    summ = pd.DataFrame({"metric": [c[6:] for c in mcols],
                         "label": [MET.METRIC_LABELS.get(c[6:], c[6:]) for c in mcols],
                         "mean": [df[c].mean() for c in mcols],
                         "sd": [df[c].std(ddof=1) for c in mcols]})
    settings = {"outer": type(outer_cv).__name__, "n_outer": len(outer),
                "inner_folds": val_cfg.inner_folds, "n_iter": n_iter, "models": model_keys,
                "primary_metric": pm, "selection_metric": sel_metric}
    return NestedCVResult(df, summ, df["selected_model"].value_counts(), feats, settings, outer)


# --------------------------------------------------------------------------- external
@dataclass
class CompatibilityReport:
    missing_features: list
    extra_columns: list
    unseen_levels: dict
    dtype_problems: list
    target_problems: list
    ok: bool

    def messages(self) -> list[str]:
        m = []
        if self.missing_features:
            m.append(f"{len(self.missing_features)} model feature(s) are missing from the "
                     f"external data: {self.missing_features[:10]}")
        if self.extra_columns:
            m.append(f"{len(self.extra_columns)} extra column(s) will be ignored.")
        for c, lv in self.unseen_levels.items():
            m.append(f"'{c}' has level(s) unseen in training: {lv[:5]} (encoded as all-zeros).")
        m += self.dtype_problems + self.target_problems
        return m


def check_compatibility(train_X: pd.DataFrame, numeric, categorical, ext: pd.DataFrame,
                        design: StudyDesign, classes) -> CompatibilityReport:
    missing = [c for c in train_X.columns if c not in ext.columns]
    extra = [c for c in ext.columns if c not in train_X.columns and c not in
             (design.target, design.sample_id, design.group, design.batch, design.time)]
    unseen, dtp, tp = {}, [], []
    for c in categorical:
        if c in ext:
            lv = set(ext[c].dropna().astype(str)) - set(train_X[c].dropna().astype(str))
            if lv:
                unseen[c] = sorted(lv)
    for c in numeric:
        if c in ext and not pd.api.types.is_numeric_dtype(ext[c]):
            dtp.append(f"'{c}' is numeric in training but not in the external data.")
    if design.target in ext:
        if classes is not None:
            new = set(ext[design.target].dropna().astype(str)) - set(classes)
            if new:
                tp.append(f"External target has unknown class(es): {sorted(new)}")
    else:
        tp.append(f"Target column '{design.target}' is absent: only predictions can be made.")
    ok = not missing and not dtp and not any("unknown class" in t for t in tp)
    return CompatibilityReport(missing, extra, unseen, dtp, tp, ok)


def dataset_shift(train_X: pd.DataFrame, ext_X: pd.DataFrame, numeric: list[str],
                  random_state: int = 0) -> dict:
    """Per-feature standardised mean difference + KS, and a domain-classifier AUC
    (how well a new, separate model can tell development from external samples)."""
    from scipy.stats import ks_2samp
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    rows = []
    for c in numeric:
        a, b = train_X[c].dropna().astype(float), ext_X[c].dropna().astype(float)
        if len(a) < 3 or len(b) < 3:
            continue
        sd = a.std(ddof=1) or 1.0
        rows.append({"feature": c, "smd": (b.mean() - a.mean()) / sd,
                     "ks": ks_2samp(a, b).statistic, "ks_p": ks_2samp(a, b).pvalue,
                     "missing_dev": train_X[c].isna().mean(), "missing_ext": ext_X[c].isna().mean()})
    tab = pd.DataFrame(rows)
    if len(tab):
        tab["abs_smd"] = tab["smd"].abs()
        tab = tab.sort_values("abs_smd", ascending=False).reset_index(drop=True)
    dom_auc = np.nan
    if numeric:
        Z = pd.concat([train_X[numeric], ext_X[numeric]], ignore_index=True).astype(float)
        lab = np.r_[np.zeros(len(train_X)), np.ones(len(ext_X))]
        clf = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                            LogisticRegression(C=0.1, max_iter=2000))
        k = int(min(5, lab.sum(), len(lab) - lab.sum()))
        if k >= 2:
            dom_auc = float(cross_val_score(clf, Z, lab, cv=k, scoring="roc_auc").mean())
    return {"features": tab, "domain_auc": dom_auc,
            "n_shifted_smd_gt_0.5": int((tab["abs_smd"] > 0.5).sum()) if len(tab) else 0}


@dataclass
class ExternalResult:
    compatibility: CompatibilityReport
    metrics: Optional[pd.DataFrame]
    predictions: pd.DataFrame
    shift: dict


def external_validate(pipe, dev: PreparedData, ext_df: pd.DataFrame, design: StudyDesign,
                      task: str, n_boot: int = 1000, random_state: int = 0,
                      allow_missing_features: bool = False) -> ExternalResult:
    """Evaluate a *fitted* pipeline on an independent cohort. Never refits ``pipe``."""
    comp = check_compatibility(dev.X, dev.numeric, dev.categorical, ext_df, design, dev.classes)
    if comp.missing_features and not allow_missing_features:
        raise ValueError("External data lack model features: " + ", ".join(comp.missing_features[:20])
                         + ". Map/rename them, or explicitly allow imputation of missing features.")
    if comp.dtype_problems:
        raise ValueError(" ".join(comp.dtype_problems))
    ext = ext_df.copy()
    for c in comp.missing_features:
        ext[c] = np.nan
    has_y = design.target in ext and ext[design.target].notna().any() and \
        not any("unknown class" in t for t in comp.target_problems)
    if has_y:
        ed = prepare_xy(ext.loc[ext[design.target].notna()], design)
        if dev.classes is not None:      # use the development label coding
            ed.y = np.array([dev.classes.index(v) for v in
                             ext.loc[ext[design.target].notna(), design.target].astype(str)])
        Xe, ye, ge = ed.X[dev.X.columns], ed.y, ed.groups
    else:
        Xe = ext[dev.X.columns].copy()
        for c in dev.categorical:
            Xe[c] = Xe[c].map(lambda v: v if pd.isna(v) else str(v))
        ye, ge = None, None
    y_pred, proba = predict_scores(pipe, Xe, task)
    pred = pd.DataFrame(index=Xe.index)
    if design.sample_id and design.sample_id in ext:
        pred[design.sample_id] = ext.loc[Xe.index, design.sample_id].values
    if task == "regression":
        pred["prediction"] = y_pred
    else:
        pred["predicted_class"] = [dev.classes[i] for i in y_pred]
        if proba is not None:
            for j, c in enumerate(dev.classes):
                pred[f"P({c})"] = proba[:, j]
    if ye is not None:
        pred["observed"] = [dev.classes[i] for i in ye] if dev.classes else ye
    met = None
    if ye is not None:
        nc = len(dev.classes) if dev.classes else 2
        met = bootstrap_ci(task, ye, y_pred, proba, nc, ge, n_boot, random_state=random_state,
                           calibrated=MET.has_proba(pipe))
    shift = dataset_shift(dev.X, Xe, dev.numeric, random_state)
    return ExternalResult(comp, met, pred, shift)


def cv_predictions(data: PreparedData, idx, task, model_key, prep_cfg, val_cfg, params=None):
    """Out-of-fold predictions on development rows (for CV plots)."""
    idx = np.asarray(idx)
    X, y = data.X.iloc[idx], data.y[idx]
    groups = data.groups[idx] if data.groups is not None else None
    cv = make_cv(inner_config(val_cfg), task, groups is not None)
    spec = registry(task).get(model_key)
    base = build_pipeline(spec.factory(val_cfg.random_state), data.numeric, data.categorical,
                          prep_cfg, task, val_cfg.random_state)
    if params:
        base.set_params(**params)
    nc = len(data.classes) if data.classes else 1
    oof_pred = np.zeros(len(idx)) if task == "regression" else np.zeros(len(idx), int)
    oof_proba = None if task == "regression" else np.zeros((len(idx), nc))
    got = False
    for tr, te in cv.split(np.zeros(len(idx)), y, groups):
        p = clone(base).fit(X.iloc[tr], y[tr])
        yp, pr = predict_scores(p, X.iloc[te], task)
        oof_pred[te] = yp
        if pr is not None:
            oof_proba[te] = pr
            got = True
    return y, oof_pred, (oof_proba if got else None)


def learning_curve_scores(data: PreparedData, idx, task, model_key, prep_cfg, val_cfg, params=None,
                          n_points: int = 6, metric: Optional[str] = None):
    """Training / CV score versus training-set size (development rows only, same CV design)."""
    from sklearn.model_selection import learning_curve
    idx = np.asarray(idx)
    X, y = data.X.iloc[idx], data.y[idx]
    groups = data.groups[idx] if data.groups is not None else None
    cv = make_cv(inner_config(val_cfg), task, groups is not None)
    spec = registry(task).get(model_key)
    pipe = build_pipeline(spec.factory(val_cfg.random_state), data.numeric, data.categorical,
                          prep_cfg, task, val_cfg.random_state)
    if params:
        pipe.set_params(**params)
    nc = len(data.classes) if data.classes else 2
    sc = MET.scorers_for(task, nc, y, spec.proba)
    metric = metric if metric in sc else (MET.primary_metric(task) if MET.primary_metric(task) in sc
                                          else "balanced_accuracy")
    sizes, tr_s, va_s = learning_curve(pipe, X, y, groups=groups, cv=cv, scoring=sc[metric],
                                       train_sizes=np.linspace(0.2, 1.0, n_points), n_jobs=resolve_jobs(val_cfg.n_jobs),
                                       error_score=np.nan, shuffle=True, random_state=val_cfg.random_state)
    if metric in MET.LOWER_IS_BETTER:
        tr_s, va_s = -tr_s, -va_s
    return sizes, tr_s, va_s, metric
