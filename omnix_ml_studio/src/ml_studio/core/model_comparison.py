"""Cross-validated model comparison on the development (non-holdout) data.

All models are evaluated on *identical* splits (the split indices are generated once
and reused), so the leaderboard compares like with like. Each result carries a
``cv_signature``; results with different signatures must not be merged.
"""
from __future__ import annotations

import hashlib
import time
import traceback
import warnings
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate

from . import classification, regression
from .config import PreprocessConfig, ValidationConfig
from .metrics import LOWER_IS_BETTER, METRIC_LABELS, primary_metric, scorers, scorers_for
from .preprocessing import PreparedData, build_pipeline
from .splitting import describe_cv, make_cv


def registry(task: str):
    return regression if task == "regression" else classification


@dataclass
class ComparisonResult:
    leaderboard: pd.DataFrame
    folds: pd.DataFrame
    settings: dict
    cv_signature: str
    failures: dict = field(default_factory=dict)


def split_signature(splits) -> str:
    h = hashlib.sha256()
    for _tr, te in splits:
        h.update(np.asarray(te, np.int64).tobytes()); h.update(b"|")
    return h.hexdigest()[:16]


def fixed_splits(data: PreparedData, idx: np.ndarray, task: str, val_cfg: ValidationConfig):
    """Materialise CV splits (positions relative to ``idx``)."""
    groups = data.groups[idx] if data.groups is not None else None
    cv = make_cv(val_cfg, task, groups is not None)
    splits = list(cv.split(np.zeros(len(idx)), data.y[idx], groups))
    return splits, describe_cv(cv, groups)


def _cv_one(payload: dict, key: str) -> tuple[dict, list]:
    """Cross-validate one model on the shared splits. Module-level so isolated workers can run it."""
    X, y, splits, task = payload["X"], payload["y"], payload["splits"], payload["task"]
    prep_cfg, val_cfg, n_classes, sc_all = payload["prep"], payload["val"], payload["n_classes"], payload["sc_all"]
    numeric, categorical = payload["numeric"], payload["categorical"]
    spec = registry(task).get(key)
    pipe = build_pipeline(spec.factory(val_cfg.random_state), numeric, categorical, prep_cfg, task,
                          val_cfg.random_state)
    sc = scorers_for(task, n_classes, y, spec.proba)
    t0 = time.time()
    res = cross_validate(pipe, X, y, cv=splits, scoring=sc, n_jobs=payload.get("inner_jobs", 1),
                         error_score="raise", return_train_score=False)
    row = {"key": key, "Model": spec.name, "family": spec.family, "probabilities": spec.proba}
    fold_rows = []
    for m in sc_all:
        if m not in sc:
            row[f"{m}_mean"] = np.nan
            row[f"{m}_sd"] = np.nan
            continue
        v = np.asarray(res[f"test_{m}"], dtype=float)
        if m in LOWER_IS_BETTER:
            v = -v
        row[f"{m}_mean"] = np.nanmean(v)
        row[f"{m}_sd"] = np.nanstd(v, ddof=1) if len(v) > 1 else np.nan
        for f, val in enumerate(v):
            fold_rows.append({"key": key, "fold": f, "metric": m, "value": val})
    row["fit_time_s"] = float(np.sum(res["fit_time"]))
    row["wall_time_s"] = time.time() - t0
    return row, fold_rows


def model_maker(spec, prep_cfg: PreprocessConfig, random_state: int, params: Optional[dict] = None):
    """Factory for the bare model as the pipeline would configure it (class weights, params)."""
    bare = {k.replace("model__", "", 1): v for k, v in (params or {}).items() if k.startswith("model__")}

    def make():
        est = spec.factory(random_state)
        if prep_cfg.imbalance == "class_weight" and "class_weight" in est.get_params():
            est.set_params(class_weight="balanced")
        if bare:
            est.set_params(**bare)
        return est
    return make


def _cv_one(payload, key, threads=1, max_seconds=None):
    """Worker task: cross-validate model ``key`` (on the fold cache when present)."""
    from .compute import Skipped, cv_on_cache
    task, prep, val = payload["task"], payload["prep"], payload["val"]
    spec = registry(task).get(key)
    y = payload["y"]
    n_classes = payload["n_classes"]
    sc = scorers_for(task, n_classes, y, spec.proba)
    cache = payload.get("cache")
    if cache is not None:
        try:
            vals, ftimes, _ = cv_on_cache(cache, model_maker(spec, prep, val.random_state), payload.get("sampler"), sc,
                                          LOWER_IS_BETTER, threads, max_seconds)
        except Skipped as e:
            return {"skipped": str(e)}
        res = {f"test_{m}": np.asarray([-v if m in LOWER_IS_BETTER else v for v in vals[m]]) for m in sc}
        res["fit_time"] = np.asarray(ftimes)
        return res
    data, idx = payload["data"], payload["idx"]
    pipe = build_pipeline(spec.factory(val.random_state), data.numeric, data.categorical, prep, task, val.random_state)
    if threads > 1:
        from .compute import set_threads
        set_threads(pipe.named_steps["model"], threads)
    return cross_validate(pipe, data.X.iloc[idx], data.y[idx], cv=payload["splits"], scoring=sc, n_jobs=1,
                          error_score="raise", return_train_score=False)


def compare_models(data: PreparedData, task: str, model_keys: list[str],
                   prep_cfg: PreprocessConfig, val_cfg: ValidationConfig,
                   train_idx: Optional[np.ndarray] = None,
                   time_budget_s: Optional[float] = None,
                   progress: Optional[Callable[[int, int, str], None]] = None,
                   isolate: bool = False, per_model_timeout_s: Optional[float] = None,
                   workers: Optional[int] = None, mem_limit_mb: Optional[int] = None,
                   should_stop: Optional[Callable[[], bool]] = None,
                   on_result: Optional[Callable[[dict], None]] = None,
                   use_cache: bool = True) -> ComparisonResult:
    """Cross-validate every model on identical folds of ``train_idx``.

    Preprocessing is fitted once per training fold and the result shared by every model (the
    *fold cache*; leakage-safe, identical to refitting it per model). ``isolate=True`` runs each
    model in a separate worker process, in parallel, with a per-model time limit and a memory
    limit: a model that exceeds either is stopped and reported, and the caller never crashes.
    A model whose first fold projects past the time limit is skipped early. ``should_stop`` stops
    the run between folds/models; ``on_result`` receives each finished leaderboard row.
    """
    from . import runner
    from .compute import Skipped, _prep_part, build_fold_cache, cv_on_cache, n_cores
    idx = np.arange(len(data.y)) if train_idx is None else np.asarray(train_idx)
    y = data.y[idx]
    splits, cv_desc = fixed_splits(data, idx, task, val_cfg)
    n_classes = len(data.classes) if data.classes else 2
    sc_all = scorers(task, n_classes, y)
    reg = registry(task)
    rows, fold_rows, failures = [], [], {}
    t_start = time.time()
    cache = None
    if use_cache:
        if progress:
            progress(0, len(model_keys), "preprocessing the folds")
        try:
            cache = build_fold_cache(data, idx, task, splits, prep_cfg, val_cfg.random_state, should_stop=should_stop)
        except Exception:  # noqa: BLE001 - fall back to per-model pipelines
            cache = None
    sampler = _prep_part(data, prep_cfg, task, val_cfg.random_state)[1] if cache is not None else None
    cores = n_cores()

    def add_row(key, res, wall):
        spec = reg.get(key)
        row = {"key": key, "Model": spec.name, "family": spec.family, "probabilities": spec.proba}
        for m in sc_all:
            if f"test_{m}" not in res:
                row[f"{m}_mean"] = np.nan
                row[f"{m}_sd"] = np.nan
                continue
            v = np.asarray(res[f"test_{m}"], dtype=float)
            if m in LOWER_IS_BETTER:
                v = -v
            with warnings.catch_warnings():            # a metric undefined on every fold stays NaN
                warnings.simplefilter("ignore", RuntimeWarning)
                row[f"{m}_mean"] = np.nanmean(v)
                row[f"{m}_sd"] = np.nanstd(v, ddof=1) if len(v) > 1 else np.nan
            for f, val in enumerate(v):
                fold_rows.append({"key": key, "fold": f, "metric": m, "value": val})
        row["fit_time_s"] = float(np.sum(res["fit_time"]))
        row["wall_time_s"] = wall
        rows.append(row)
        if on_result:
            on_result(dict(row))

    if isolate:
        n_w, auto_mem = runner.plan_workers(workers)
        n_w = max(1, min(n_w, len(model_keys)))
        threads = max(1, cores // n_w)
        payload = {"cache": cache, "sampler": sampler, "task": task, "prep": prep_cfg, "val": val_cfg, "y": y,
                   "n_classes": n_classes, "splits": splits,
                   "data": None if cache is not None else data, "idx": idx}
        tasks = [(k, {"key": k, "threads": threads, "max_seconds": per_model_timeout_s}) for k in model_keys]

        def done(tid, r):
            if r.status == "ok" and isinstance(r.value, dict) and "skipped" not in r.value:
                add_row(tid, r.value, r.seconds)

        out = runner.run_isolated(payload, _cv_one, tasks, workers=n_w, timeout_s=per_model_timeout_s,
                                  mem_limit_mb=mem_limit_mb or auto_mem, total_budget_s=time_budget_s,
                                  progress=progress, should_stop=should_stop, on_done=done)
        for k in model_keys:
            r = out.get(k)
            if r is None:
                failures[k] = "not run"
            elif r.status == "ok" and isinstance(r.value, dict) and "skipped" in r.value:
                failures[k] = f"skipped: {r.value['skipped']}"
            elif r.status != "ok":
                failures[k] = r.message.split("\n")[0]
    else:
        for i, key in enumerate(model_keys):
            spec = reg.get(key)
            if should_stop and should_stop():
                failures[key] = "skipped: stopped by user"
                continue
            if progress:
                progress(i, len(model_keys), spec.name)
            if time_budget_s and time.time() - t_start > time_budget_s:
                failures[key] = "skipped: time budget exhausted"
                continue
            sc = scorers_for(task, n_classes, y, spec.proba)
            t0 = time.time()
            try:
                if cache is not None:
                    vals, ftimes, _ = cv_on_cache(cache, model_maker(spec, prep_cfg, val_cfg.random_state), sampler,
                                                  sc, LOWER_IS_BETTER, cores, per_model_timeout_s, should_stop)
                    res = {f"test_{m}": np.asarray([-v if m in LOWER_IS_BETTER else v for v in vals[m]]) for m in sc}
                    res["fit_time"] = np.asarray(ftimes)
                else:
                    pipe = build_pipeline(spec.factory(val_cfg.random_state), data.numeric, data.categorical,
                                          prep_cfg, task, val_cfg.random_state)
                    res = cross_validate(pipe, data.X.iloc[idx], y, cv=splits, scoring=sc,
                                         n_jobs=runner.resolve_jobs(val_cfg.n_jobs) if val_cfg.n_jobs != 1 else 1,
                                         error_score="raise", return_train_score=False)
            except Skipped as e:
                failures[key] = f"skipped: {e}"
                continue
            except MemoryError:
                failures[key] = "out of memory"
                continue
            except Exception as e:  # noqa: BLE001 - report and continue with other models
                failures[key] = f"{type(e).__name__}: {e}"
                failures[key + "__trace"] = traceback.format_exc(limit=3)
                continue
            add_row(key, res, time.time() - t0)
    lb = pd.DataFrame(rows)
    pm = primary_metric(task)
    if len(lb):
        lb = lb.sort_values(f"{pm}_mean", ascending=pm in LOWER_IS_BETTER,
                            na_position="last").reset_index(drop=True)
        lb.insert(0, "rank", np.arange(1, len(lb) + 1))
        if "dummy" in set(lb["key"]):
            base = lb.loc[lb["key"] == "dummy", f"{pm}_mean"].iloc[0]
            lb["beats_baseline"] = (lb[f"{pm}_mean"] > base) if pm not in LOWER_IS_BETTER \
                else (lb[f"{pm}_mean"] < base)
    settings = {"task": task, "cv": cv_desc, "n_splits": len(splits), "n_train": len(idx),
                "primary_metric": METRIC_LABELS.get(pm, pm),
                "preprocessing": prep_cfg.to_dict(), "validation": val_cfg.to_dict(),
                "models": model_keys, "isolated": isolate, "threads": cores,
                "fold_cache": None if cache is None else
                {"dtype": cache.dtype, "mb": round(cache.mb, 1), "seconds": round(cache.seconds, 1),
                 "features_after_preprocessing": int(np.median(cache.n_features_out))},
                "total_seconds": round(time.time() - t_start, 1)}
    return ComparisonResult(lb, pd.DataFrame(fold_rows), settings, split_signature(splits),
                            {k: v for k, v in failures.items() if not k.endswith("__trace")})


def pretty_leaderboard(lb: pd.DataFrame, task: str) -> pd.DataFrame:
    """Mean ± SD strings with readable metric names."""
    if lb.empty:
        return lb
    cols = ["rank", "Model"]
    out = lb[cols].copy()
    for c in lb.columns:
        if c.endswith("_mean"):
            m = c[:-5]
            out[METRIC_LABELS.get(m, m)] = ["—" if not np.isfinite(a) else
                                            (f"{a:.3f} ± {b:.3f}" if np.isfinite(b) else f"{a:.3f}")
                                            for a, b in zip(lb[c], lb[f"{m}_sd"])]
    out["Fit time (s)"] = lb["fit_time_s"].round(1)
    if "beats_baseline" in lb:
        out["Beats dummy"] = lb["beats_baseline"].map({True: "yes", False: "NO"})
    return out


@dataclass
class CreatedModel:
    key: str
    name: str
    params: dict
    folds: pd.DataFrame          # one row per CV fold, one column per metric (+ Mean / SD rows)
    summary: dict                # metric -> (mean, sd)
    how: str
    cv_signature: str


def create_model(data: PreparedData, idx: np.ndarray, task: str, model_key: str,
                 prep_cfg: PreprocessConfig, val_cfg: ValidationConfig, params: Optional[dict] = None,
                 how: str = "default hyperparameters", should_stop: Optional[Callable[[], bool]] = None) -> CreatedModel:
    """Cross-validate one model (optionally with hyperparameters) on the same folds as
    :func:`compare_models`, returning the per-fold score table."""
    from .compute import _prep_part, build_fold_cache, cv_on_cache, n_cores
    idx = np.asarray(idx)
    X, y = data.X.iloc[idx], data.y[idx]
    splits, _ = fixed_splits(data, idx, task, val_cfg)
    n_classes = len(data.classes) if data.classes else 2
    spec = registry(task).get(model_key)
    sc = scorers_for(task, n_classes, y, spec.proba)
    try:                       # fold cache: multithreaded and memory-guarded, whatever n_jobs says
        cache = build_fold_cache(data, idx, task, splits, prep_cfg, val_cfg.random_state, should_stop=should_stop)
    except Exception:  # noqa: BLE001 - fall back to a per-fold pipeline
        cache = None
    if cache is not None:
        sampler = _prep_part(data, prep_cfg, task, val_cfg.random_state)[1]
        vals, _, _ = cv_on_cache(cache, model_maker(spec, prep_cfg, val_cfg.random_state, params), sampler, sc,
                                 LOWER_IS_BETTER, n_cores(), None, should_stop)
        res = {f"test_{m}": np.asarray([-v if m in LOWER_IS_BETTER else v for v in vals[m]]) for m in sc}
    else:
        pipe = build_pipeline(spec.factory(val_cfg.random_state), data.numeric, data.categorical,
                              prep_cfg, task, val_cfg.random_state)
        if params:
            pipe.set_params(**params)
        res = cross_validate(pipe, X, y, cv=splits, scoring=sc, n_jobs=val_cfg.n_jobs, error_score="raise")
    tab = {}
    for m in sc:
        v = np.asarray(res[f"test_{m}"], dtype=float)
        tab[METRIC_LABELS.get(m, m)] = -v if m in LOWER_IS_BETTER else v
    folds = pd.DataFrame(tab)
    folds.index = [f"Fold {i + 1}" for i in range(len(folds))]
    summary = {m: (float(folds[METRIC_LABELS.get(m, m)].mean()), float(folds[METRIC_LABELS.get(m, m)].std(ddof=1)))
               for m in sc}
    folds = pd.concat([folds, folds.mean().to_frame("Mean").T, folds.std(ddof=1).to_frame("SD").T])
    return CreatedModel(model_key, spec.name, dict(params or {}), folds, summary, how, split_signature(splits))
