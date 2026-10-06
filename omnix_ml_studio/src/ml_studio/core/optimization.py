"""Hyperparameter optimisation on development data only (randomised search).

The search runs on ``idx`` (training / outer-training rows); CV splits inside the
search are group-aware when groups exist. Holdout rows are never passed in.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, cross_val_score

from .config import PreprocessConfig, ValidationConfig
from .runner import resolve_jobs
from .metrics import primary_metric, scorers_for
from .model_comparison import registry
from .preprocessing import PreparedData, build_pipeline
from .splitting import make_cv

_BASE = {"repeated_stratified_kfold": "stratified_kfold", "repeated_kfold": "kfold"}


@dataclass
class TuningResult:
    model_key: str
    best_params: dict
    best_score: float
    metric: str
    cv_results: pd.DataFrame
    estimator: object            # refit on all of ``idx``
    n_iter: int


def inner_config(val_cfg: ValidationConfig, n_folds: Optional[int] = None) -> ValidationConfig:
    """Non-repeated version of the strategy for inner loops (keeps cost bounded)."""
    return replace(val_cfg, cv_strategy=_BASE.get(val_cfg.cv_strategy, val_cfg.cv_strategy),
                   n_repeats=1, n_folds=n_folds or val_cfg.n_folds)


def tune(data: PreparedData, idx: np.ndarray, task: str, model_key: str,
         prep_cfg: PreprocessConfig, val_cfg: ValidationConfig, n_iter: int = 20,
         metric: Optional[str] = None, n_folds: Optional[int] = None,
         random_state: Optional[int] = None, use_cache: bool = True,
         progress=None, should_stop=None) -> TuningResult:
    """Random search over the model's hyperparameters, scored by CV on ``idx``.

    Fast path: preprocessing is fitted once per fold and reused by every setting (identical
    settings to ``RandomizedSearchCV`` with the same seed; leakage-safe because each fold's
    preprocessing still sees only that fold's training rows)."""
    idx = np.asarray(idx)
    X, y = data.X.iloc[idx], data.y[idx]
    groups = data.groups[idx] if data.groups is not None else None
    rs = val_cfg.random_state if random_state is None else random_state
    cfg = inner_config(val_cfg, n_folds)
    splits = list(make_cv(cfg, task, groups is not None, random_state=rs)
                  .split(np.zeros(len(idx)), y, groups))
    n_classes = len(data.classes) if data.classes else 2
    spec = registry(task).get(model_key)
    metric = metric or primary_metric(task)
    avail = scorers_for(task, n_classes, y, spec.proba)
    if metric not in avail:                    # e.g. multiclass AUC for a model without probabilities
        metric = "balanced_accuracy"
    scorer = avail[metric]
    pipe = build_pipeline(spec.factory(rs), data.numeric, data.categorical, prep_cfg, task, rs)
    if not spec.space:
        s = cross_val_score(pipe, X, y, cv=splits, scoring=scorer, n_jobs=resolve_jobs(val_cfg.n_jobs))
        pipe.fit(X, y)
        return TuningResult(model_key, {}, float(np.mean(s)), metric,
                            pd.DataFrame({"mean_test_score": [np.mean(s)]}), pipe, 0)
    if use_cache and all(k.startswith("model__") for k in spec.space):
        fast = _tune_cached(data, idx, task, spec, prep_cfg, val_cfg, splits, scorer, metric, n_iter, rs,
                            progress, should_stop)
        if fast is not None:
            return fast
    search = RandomizedSearchCV(pipe, spec.space, n_iter=n_iter, scoring=scorer, cv=splits,
                                random_state=rs, n_jobs=resolve_jobs(val_cfg.n_jobs), refit=True,
                                error_score=np.nan)
    search.fit(X, y)
    cvr = pd.DataFrame(search.cv_results_)
    keep = [c for c in cvr.columns if c.startswith("param_")] + ["mean_test_score", "std_test_score",
                                                                 "rank_test_score", "mean_fit_time"]
    return TuningResult(model_key, search.best_params_, float(search.best_score_), metric,
                        cvr[keep].sort_values("rank_test_score"), search.best_estimator_, n_iter)


def _tune_cached(data, idx, task, spec, prep_cfg, val_cfg, splits, scorer, metric, n_iter, rs,
                 progress=None, should_stop=None) -> Optional[TuningResult]:
    import time

    from sklearn.model_selection import ParameterSampler

    from .compute import Skipped, _prep_part, build_fold_cache, cv_on_cache, n_cores
    from .model_comparison import model_maker
    cache = build_fold_cache(data, idx, task, splits, prep_cfg, rs, should_stop=should_stop)
    if cache is None:
        return None
    sampler = _prep_part(data, prep_cfg, task, rs)[1]
    settings = list(ParameterSampler(spec.space, n_iter, random_state=rs))
    rows, cores = [], n_cores()
    for i, params in enumerate(settings):
        if should_stop and should_stop():
            break
        if progress:
            progress(i, len(settings), f"setting {i + 1} of {len(settings)}")
        t = time.time()
        try:
            vals, ft, _ = cv_on_cache(cache, model_maker(spec, prep_cfg, rs, params), sampler, {metric: scorer},
                                      (), cores, None, should_stop)
            v = np.asarray(vals[metric], float)
        except Skipped:
            break
        except Exception:  # noqa: BLE001 - same as error_score=np.nan
            v, ft = np.full(len(splits), np.nan), [time.time() - t]
        rows.append({**{f"param_{k}": p for k, p in params.items()}, "mean_test_score": np.nanmean(v) if
                     np.isfinite(v).any() else np.nan, "std_test_score": np.nanstd(v) if np.isfinite(v).any()
                     else np.nan, "mean_fit_time": float(np.mean(ft)), "_params": params})
    ok = [r for r in rows if np.isfinite(r["mean_test_score"])]
    if not ok:
        return None
    cvr = pd.DataFrame(rows)
    cvr["rank_test_score"] = cvr["mean_test_score"].rank(ascending=False, method="min").fillna(len(cvr)).astype(int)
    best = max(ok, key=lambda r: r["mean_test_score"])
    X, y = data.X.iloc[idx], data.y[idx]
    est = build_pipeline(spec.factory(rs), data.numeric, data.categorical, prep_cfg, task, rs)
    est.set_params(**best["_params"])
    est.fit(X, y)
    keep = [c for c in cvr.columns if c.startswith("param_")] + ["mean_test_score", "std_test_score",
                                                                 "rank_test_score", "mean_fit_time"]
    return TuningResult(spec.key, dict(best["_params"]), float(best["mean_test_score"]), metric,
                        cvr[keep].sort_values("rank_test_score"), est, len(rows))
