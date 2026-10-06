"""Module-level task functions for :mod:`runner` (each receives the shared payload first).

Payload keys: data (PreparedData), idx (development rows), task, prep, val.
"""
from __future__ import annotations

from dataclasses import replace

from .runner import resolve_jobs


def _val(p):
    return replace(p["val"], n_jobs=resolve_jobs(p["val"].n_jobs))


def create(p, key, params=None, how="default hyperparameters"):
    from .model_comparison import create_model
    return create_model(p["data"], p["idx"], p["task"], key, p["prep"], _val(p), params, how)


def tune(p, key, n_iter=20, metric=None):
    from .optimization import tune as _tune
    return _tune(p["data"], p["idx"], p["task"], key, p["prep"], _val(p), n_iter=n_iter, metric=metric)


def nested(p, keys, n_iter=10):
    from .validation import nested_cv
    return nested_cv(p["data"], p["idx"], p["task"], keys, p["prep"], _val(p), n_iter=n_iter)


def oof(p, key, params=None):
    from .validation import cv_predictions
    return cv_predictions(p["data"], p["idx"], p["task"], key, p["prep"], _val(p), params)


def learning_curve(p, key, params=None, n_points=6):
    from .validation import learning_curve_scores
    return learning_curve_scores(p["data"], p["idx"], p["task"], key, p["prep"], _val(p), params, n_points)


def fit(p, key, params=None, idx=None):
    from .validation import fit_final
    return fit_final(p["data"], p["idx"] if idx is None else idx, p["task"], key, p["prep"], p["val"], params)
