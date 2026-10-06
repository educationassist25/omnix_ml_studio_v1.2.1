"""Compute resources and fast, leakage-safe cross-validation.

Speed comes from three things, none of which changes what is estimated:

1. **Fold cache.** Preprocessing (filters, imputation, scaling, in-fold feature selection) is
   fitted on each training fold exactly once and its output reused by every model and every
   hyperparameter setting. Each fold's preprocessing still sees only that fold's training rows,
   so this is the same procedure a per-model pipeline runs, without repeating it 20 or 2,000 times.
2. **All CPU cores.** Libraries with native threading (LightGBM, XGBoost, CatBoost, random
   forests, KNN) get every core; other models run their folds in parallel threads (no data copies).
3. **Memory guard.** The cache is stored as float32 when large and is skipped entirely when it
   would not fit comfortably in the available memory, falling back to per-model pipelines.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from sklearn.base import clone

THREAD_PARAMS = ("n_jobs", "thread_count", "nthread")


# --------------------------------------------------------------------------- resources
def n_cores() -> int:
    """CPU cores this process may use (container quota and affinity aware; ``OMNIX_MAX_THREADS`` caps it)."""
    from .runner import available_cpus
    n = available_cpus()
    env = os.environ.get("OMNIX_MAX_THREADS")
    if env and env.isdigit():
        n = min(n, max(1, int(env)))
    return max(1, n)


def available_memory_mb() -> float:
    from .runner import available_memory_mb as _mem
    return float(_mem())


# ``n_jobs`` of these starts *processes* (loky) and copies or memory-maps the data for every fit:
# with wide data that is far slower than one thread, and leaves temporary folders behind. Their
# folds are run in parallel threads instead.
PROCESS_PARALLEL = {"LogisticRegression", "LinearRegression", "TheilSenRegressor", "GaussianProcessClassifier",
                    "ScaledGPC"}


def set_threads(est, n: int):
    """Give an estimator ``n`` native threads where it supports them. Returns True if it does."""
    params = est.get_params(deep=False)
    if type(est).__name__ in PROCESS_PARALLEL:
        if "n_jobs" in params:
            est.set_params(n_jobs=1)
        return False
    hit = False
    for p in THREAD_PARAMS:
        if p in params:
            est.set_params(**{p: n})
            hit = True
    return hit


# --------------------------------------------------------------------------- fold cache
@dataclass
class FoldCache:
    folds: list            # [(Xtr, ytr, Xte, yte)] preprocessed, numpy
    dtype: str
    n_features_out: list
    seconds: float
    mb: float
    columns: list = None   # model-input column names per fold


def _names_out(P, k):
    try:
        return list(P.get_feature_names_out())
    except Exception:  # noqa: BLE001
        return [f"x{i}" for i in range(k)]


def _framed(est, X, cols):
    """Models that read features by name (``_omnix_wants_frame``) get a labelled frame; others the array."""
    if cols is not None and getattr(est, "_omnix_wants_frame", False):
        import pandas as pd
        return pd.DataFrame(X, columns=cols, copy=False)
    return X


def _prep_part(data, prep_cfg, task, random_state):
    """The preprocessing steps of the modelling pipeline (everything before resampling and the model)."""
    from sklearn.dummy import DummyClassifier, DummyRegressor
    from sklearn.pipeline import Pipeline

    from .preprocessing import build_pipeline
    base = DummyRegressor() if task == "regression" else DummyClassifier()
    full = build_pipeline(base, data.numeric, data.categorical, prep_cfg, task, random_state)
    steps = [(n, s) for n, s in full.steps if n not in ("resample", "model")]
    sampler = dict(full.steps).get("resample")
    return Pipeline(steps) if steps else None, sampler


def build_fold_cache(data, idx, task, splits, prep_cfg, random_state: int = 0,
                     memory_fraction: float = 0.35, should_stop: Optional[Callable] = None) -> Optional[FoldCache]:
    """Fit preprocessing once per training fold. ``None`` if the cache would not fit in memory."""
    idx = np.asarray(idx)
    X, y = data.X.iloc[idx], data.y[idx]
    n, p = X.shape
    per_fold_rows = n                                      # train + test rows of one fold
    est64 = len(splits) * per_fold_rows * max(p, 1) * 8 / 2 ** 20
    budget = memory_fraction * available_memory_mb()
    dtype = np.float32 if est64 > 256 else np.float64
    if est64 * (0.5 if dtype == np.float32 else 1.0) > budget and not prep_cfg.univariate_k and not prep_cfg.fs_method:
        return None                                        # too big and no selection to shrink it
    prep, _ = _prep_part(data, prep_cfg, task, random_state)
    t0 = time.time()
    folds, widths, mb, names = [], [], 0.0, []
    for tr, te in splits:
        if should_stop and should_stop():
            return None
        if prep is None:
            Xtr, Xte = X.iloc[tr].to_numpy(dtype), X.iloc[te].to_numpy(dtype)
            cols = list(X.columns)
        else:
            P = clone(prep)
            out = P.fit_transform(X.iloc[tr], y[tr])
            cols = list(out.columns) if hasattr(out, "columns") else _names_out(P, out.shape[1])
            Xtr = np.asarray(out, dtype=dtype)
            Xte = np.asarray(P.transform(X.iloc[te]), dtype=dtype)
        names.append(cols)
        folds.append((np.ascontiguousarray(Xtr), y[tr], np.ascontiguousarray(Xte), y[te]))
        widths.append(Xtr.shape[1])
        mb += (Xtr.nbytes + Xte.nbytes) / 2 ** 20
        if mb > budget:
            return None
    return FoldCache(folds, np.dtype(dtype).name, widths, time.time() - t0, mb, names)


def _fit_score_fold(est, sampler, Xtr, ytr, Xte, yte, scorers, lower_is_better, cols=None):
    if sampler is not None:
        Xtr, ytr = clone(sampler).fit_resample(Xtr, ytr)
    Xtr, Xte = _framed(est, Xtr, cols), _framed(est, Xte, cols)
    t = time.time()
    est.fit(Xtr, ytr)
    fit_t = time.time() - t
    out = {}
    for m, sc in scorers.items():
        try:
            v = float(sc(est, Xte, yte))
        except Exception:  # noqa: BLE001 - metric undefined on this fold
            v = np.nan
        out[m] = -v if m in lower_is_better else v
    return out, fit_t


class Skipped(Exception):
    pass


def cv_on_cache(cache: FoldCache, make_est: Callable, sampler, scorers: dict, lower_is_better=(),
                threads: Optional[int] = None, max_seconds: Optional[float] = None,
                should_stop: Optional[Callable] = None) -> tuple[dict, list, float]:
    """Cross-validate one estimator on cached folds.

    Returns ({metric: [fold values]}, fold fit times, wall seconds). Raises ``Skipped`` if the
    projected time exceeds ``max_seconds`` (decided after the first fold) or a stop is requested.
    """
    from joblib import Parallel, delayed
    from sklearn.metrics import get_scorer
    scorers = {m: (get_scorer(sc) if isinstance(sc, str) else sc) for m, sc in scorers.items()}
    threads = threads or n_cores()
    t0 = time.time()
    est0 = make_est()
    native = set_threads(est0, threads)
    vals = {m: [] for m in scorers}
    times = []
    cols = cache.columns or [None] * len(cache.folds)
    first, rest = cache.folds[0], list(zip(cache.folds[1:], cols[1:]))
    r, ft = _fit_score_fold(est0, sampler, *first, scorers, lower_is_better, cols[0])
    times.append(ft)
    for m in scorers:
        vals[m].append(r[m])
    elapsed = time.time() - t0
    if max_seconds and elapsed * len(cache.folds) > max_seconds:
        raise Skipped(f"projected {elapsed * len(cache.folds) / 60:.1f} min > limit of {max_seconds / 60:.1f} min")
    if should_stop and should_stop():
        raise Skipped("stopped")

    def run(item):
        fold, c = item
        e = make_est()
        set_threads(e, threads if native else 1)
        return _fit_score_fold(e, sampler, *fold, scorers, lower_is_better, c)

    if native or threads == 1 or len(rest) <= 1:
        res = []
        for f in rest:
            if should_stop and should_stop():
                raise Skipped("stopped")
            res.append(run(f))
    else:
        res = Parallel(n_jobs=min(threads, len(rest)), backend="threading")(delayed(run)(f) for f in rest)
    for r, ft in res:
        times.append(ft)
        for m in scorers:
            vals[m].append(r[m])
    return vals, times, time.time() - t0
