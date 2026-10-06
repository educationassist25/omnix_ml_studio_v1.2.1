"""Holdout splits and CV splitters, group-aware.

When a group (patient) column is given, every split keeps all rows of a group on the
same side. The holdout is created once, before any modelling, and its indices are
returned so the caller can keep it sealed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.model_selection import (GroupKFold, KFold, RepeatedKFold, RepeatedStratifiedKFold,
                                     StratifiedGroupKFold, StratifiedKFold, TimeSeriesSplit,
                                     train_test_split)

from .config import ValidationConfig


@dataclass
class HoldoutSplit:
    train_idx: np.ndarray        # positional indices into the prepared data
    test_idx: np.ndarray
    stratified: bool
    grouped: bool
    description: str


def holdout_split(y: np.ndarray, task: str, fraction: float = 0.2, groups=None,
                  random_state: int = 42) -> HoldoutSplit:
    n = len(y)
    idx = np.arange(n)
    strat = task != "regression"
    if fraction <= 0:
        return HoldoutSplit(idx, np.array([], int), strat, groups is not None, "no holdout")
    if groups is not None:
        k = max(2, int(round(1 / fraction)))
        if strat:
            spl = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=random_state)
            tr, te = next(spl.split(idx, y, groups))
        else:
            rng = np.random.default_rng(random_state)
            ug = np.unique(groups)
            test_g = rng.choice(ug, max(1, int(round(fraction * len(ug)))), replace=False)
            te = idx[np.isin(groups, test_g)]
            tr = idx[~np.isin(groups, test_g)]
        desc = (f"group-wise {'stratified ' if strat else ''}holdout: "
                f"{len(np.unique(groups[te]))} of {len(np.unique(groups))} groups "
                f"({len(te)} rows) held out")
        return HoldoutSplit(np.sort(tr), np.sort(te), strat, True, desc)
    tr, te = train_test_split(idx, test_size=fraction, random_state=random_state,
                              stratify=y if strat else None)
    return HoldoutSplit(np.sort(tr), np.sort(te), strat, False,
                        f"{'stratified ' if strat else ''}random holdout: {len(te)} of {n} samples "
                        f"({fraction:.0%})")


class RepeatedStratifiedGroupKFold:
    """Repeats StratifiedGroupKFold with different shuffles (not in scikit-learn)."""

    def __init__(self, n_splits=5, n_repeats=3, random_state=None, stratify=True):
        self.n_splits, self.n_repeats = n_splits, n_repeats
        self.random_state, self.stratify = random_state, stratify

    def split(self, X, y=None, groups=None):
        rng = np.random.RandomState(self.random_state)
        for _ in range(self.n_repeats):
            seed = rng.randint(0, 2 ** 31 - 1)
            if self.stratify:
                yield from StratifiedGroupKFold(self.n_splits, shuffle=True,
                                                random_state=seed).split(X, y, groups)
            else:
                ug = np.unique(groups)
                perm = np.random.RandomState(seed).permutation(ug)
                fold_of = {g: i % self.n_splits for i, g in enumerate(perm)}
                f = np.array([fold_of[g] for g in groups])
                for k in range(self.n_splits):
                    yield np.where(f != k)[0], np.where(f == k)[0]

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits * self.n_repeats


def make_cv(cfg: ValidationConfig, task: str, has_groups: bool, n_folds: Optional[int] = None,
            random_state: Optional[int] = None):
    """Return a splitter. Grouped strategies require groups to be passed to ``split``."""
    k = n_folds or cfg.n_folds
    rs = cfg.random_state if random_state is None else random_state
    s = cfg.cv_strategy
    if has_groups and s not in ("group_kfold", "stratified_group_kfold", "time_series"):
        raise ValueError(f"A group column is set but CV strategy '{s}' ignores groups — rows "
                         "from the same group would leak across folds. Use a grouped strategy.")
    if s == "stratified_kfold":
        return StratifiedKFold(k, shuffle=True, random_state=rs)
    if s == "kfold":
        return KFold(k, shuffle=True, random_state=rs)
    if s == "repeated_stratified_kfold":
        return RepeatedStratifiedKFold(n_splits=k, n_repeats=cfg.n_repeats, random_state=rs)
    if s == "repeated_kfold":
        return RepeatedKFold(n_splits=k, n_repeats=cfg.n_repeats, random_state=rs)
    if s == "group_kfold":
        if cfg.n_repeats > 1:
            return RepeatedStratifiedGroupKFold(k, cfg.n_repeats, rs, stratify=False)
        return GroupKFold(k)
    if s == "stratified_group_kfold":
        if task == "regression":
            raise ValueError("Stratified grouped CV needs a categorical target; use group_kfold.")
        if cfg.n_repeats > 1:
            return RepeatedStratifiedGroupKFold(k, cfg.n_repeats, rs, stratify=True)
        return StratifiedGroupKFold(k, shuffle=True, random_state=rs)
    if s == "time_series":
        return TimeSeriesSplit(n_splits=k)
    raise ValueError(f"Unknown CV strategy {s}")


def recommend_strategy(task: str, n: int, has_groups: bool, min_class: Optional[int] = None,
                       has_time: bool = False) -> tuple[str, int, int, str]:
    """(strategy, folds, repeats, rationale). The user must confirm it."""
    if has_groups:
        s = "stratified_group_kfold" if task != "regression" else "group_kfold"
        why = ("Repeated measures / grouped samples detected: all rows of a group must stay "
               "in the same fold to avoid leakage.")
    elif task == "regression":
        s, why = "kfold", "Regression: shuffled K-fold."
    else:
        s, why = "stratified_kfold", "Classification: stratified K-fold keeps class proportions."
    k = 5
    if min_class is not None and min_class < 10:
        k = max(2, min(5, int(min_class)))
        why += f" Folds reduced to {k} because the smallest class has {min_class} samples."
    repeats = 1
    if n < 200:
        repeats = 5
        why += " Small sample size: repeat CV 5× with different splits to stabilise estimates."
        s = {"stratified_kfold": "repeated_stratified_kfold", "kfold": "repeated_kfold"}.get(s, s)
    if has_time:
        why += (" A time column exists: if the goal is forecasting future samples, consider "
                "time-series splitting instead.")
    return s, k, repeats, why


def describe_cv(cv, groups=None) -> str:
    name = type(cv).__name__
    k = cv.get_n_splits(groups=groups) if groups is not None else cv.get_n_splits()
    return f"{name} ({k} train/validation splits{', grouped' if groups is not None else ''})"
