"""Turn multiclass predictions into a chosen comparison (one class vs rest, or class A vs class B)."""
from __future__ import annotations

import numpy as np


def one_vs_rest(y, proba, classes, cls):
    """Binary view: positive = ``cls``. Returns (y01, P with columns [rest, cls], labels, mask)."""
    k = list(classes).index(cls)
    y = np.asarray(y)
    p = proba[:, k]
    return (y == k).astype(int), np.c_[1 - p, p], [f"not {cls}", str(cls)], np.ones(len(y), bool)


def pairwise(y, proba, classes, a, b):
    """Binary view restricted to samples of class ``a`` or ``b``; positive = ``a``.

    Score = P(a) / (P(a) + P(b)), the model's preference for a over b."""
    ia, ib = list(classes).index(a), list(classes).index(b)
    y = np.asarray(y)
    mask = (y == ia) | (y == ib)
    pa, pb = proba[mask, ia], proba[mask, ib]
    s = pa / np.maximum(pa + pb, 1e-12)
    return (y[mask] == ia).astype(int), np.c_[1 - s, s], [str(b), str(a)], mask


def flip_binary(y, proba, classes):
    """Swap which class is treated as positive in a binary task."""
    return 1 - np.asarray(y), proba[:, ::-1], [classes[1], classes[0]]
