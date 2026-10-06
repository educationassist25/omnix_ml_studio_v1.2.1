"""A classifier that records which rows it sees, identified by a ``rid`` feature."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

LOG: list = []


def _ids(X):
    return np.asarray(X["rid"]).round().astype(int) if hasattr(X, "columns") else None


class Spy(ClassifierMixin, BaseEstimator):
    _omnix_wants_frame = True        # read the ``rid`` column by name on every code path (incl. the fold cache)

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        LOG.append(("fit", set(_ids(X).tolist()), len(X), np.bincount(np.asarray(y).astype(int)), id(self)))
        self.classes_ = np.unique(y)
        self.prior_ = np.bincount(np.asarray(y).astype(int), minlength=len(self.classes_)) / len(y)
        return self

    def predict_proba(self, X):
        LOG.append(("predict", set(_ids(X).tolist()), len(X), None, id(self)))
        r = (np.asarray(X["rid"]) % 7) / 1000.0
        p = np.tile(self.prior_, (len(X), 1)).astype(float)
        p[:, -1] = np.clip(p[:, -1] + r, 0, 1); p[:, 0] = 1 - p[:, 1:].sum(1)
        return p

    def predict(self, X):
        return np.argmax(self.predict_proba(X), 1)


def with_rid(df):
    out = df.copy()
    out.insert(1, "rid", np.arange(len(out)).astype(float))
    return out
