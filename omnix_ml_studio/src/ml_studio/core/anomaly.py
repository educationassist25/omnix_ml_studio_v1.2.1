"""Anomaly detection: PyCaret's detector list (same IDs), implemented with PyOD.

Scores are oriented so that *higher = more anomalous*. A detector flags the top
``contamination`` fraction of its training scores; that threshold is kept for new data.

Evaluation:
* with a reference label (known anomalies, never used for fitting): ROC-AUC, average
  precision, precision@n (n = number of known anomalies), recall among flagged;
* without labels: agreement between detectors (Spearman of scores, Jaccard of flags),
  flag stability under subsampling, and per-sample feature deviations (robust z).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd
from sklearn import metrics as M


@dataclass
class DetectorSpec:
    key: str
    name: str
    factory: Callable[[int, float, dict], object]
    params: dict = field(default_factory=dict)
    family: str = "Proximity"
    speed: str = "fast"
    notes: str = ""
    transductive: bool = False     # scores a batch only relative to itself: cannot score new samples


def _pyod(cls_path, **fixed):
    mod, cls = cls_path.rsplit(".", 1)

    def make(rs, contamination, p):
        import importlib
        import inspect
        C = getattr(importlib.import_module(mod), cls)
        kw = {**fixed, **p, "contamination": contamination}
        if "random_state" in inspect.signature(C.__init__).parameters:
            kw["random_state"] = rs
        return C(**kw)
    return make


REGISTRY: dict[str, DetectorSpec] = {s.key: s for s in [
    DetectorSpec("iforest", "Isolation Forest", _pyod("pyod.models.iforest.IForest", n_estimators=200),
                 {"n_estimators": (200, "int", 50, 1000), "max_features": (1.0, "float", 0.1, 1.0)},
                 family="Ensemble", notes="Good default; scales to many samples."),
    DetectorSpec("knn", "K-Nearest Neighbours", _pyod("pyod.models.knn.KNN"),
                 {"n_neighbors": (5, "int", 2, 50)}, family="Proximity"),
    DetectorSpec("lof", "Local Outlier Factor", _pyod("pyod.models.lof.LOF"),
                 {"n_neighbors": (20, "int", 5, 100)}, family="Proximity",
                 notes="Detects points in locally sparse regions."),
    DetectorSpec("cof", "Connectivity-Based Outlier Factor", _pyod("pyod.models.cof.COF"),
                 {"n_neighbors": (20, "int", 5, 100)}, family="Proximity", speed="slow", transductive=True,
                 notes="Scores a dataset relative to itself; cannot score new samples."),
    DetectorSpec("cluster", "Clustering-Based LOF (CBLOF)", _pyod("pyod.models.cblof.CBLOF", n_clusters=8),
                 {"n_clusters": (8, "int", 2, 30)}, family="Proximity"),
    DetectorSpec("histogram", "Histogram-Based (HBOS)", _pyod("pyod.models.hbos.HBOS"),
                 {"n_bins": (10, "int", 5, 50)}, family="Statistical",
                 notes="Assumes feature independence; very fast."),
    DetectorSpec("pca", "Principal Component Analysis", _pyod("pyod.models.pca.PCA"), {},
                 family="Linear", notes="Large reconstruction error in minor components."),
    DetectorSpec("mcd", "Minimum Covariance Determinant", _pyod("pyod.models.mcd.MCD"), {},
                 family="Linear", notes="Robust Mahalanobis distance; needs n > p."),
    DetectorSpec("svm", "One-Class SVM", _pyod("pyod.models.ocsvm.OCSVM"),
                 {"nu": (0.5, "float", 0.01, 0.99)}, family="Kernel"),
    DetectorSpec("abod", "Angle-Based (ABOD)", _pyod("pyod.models.abod.ABOD"), {},
                 family="Proximity", speed="slow", notes="Robust in high dimensions; O(n²)."),
    DetectorSpec("sod", "Subspace Outlier Detection", _pyod("pyod.models.sod.SOD"), {},
                 family="Proximity", speed="slow", transductive=True,
                 notes="Scores a dataset relative to itself; cannot score new samples."),
    DetectorSpec("sos", "Stochastic Outlier Selection", _pyod("pyod.models.sos.SOS"), {},
                 family="Statistical", speed="slow", transductive=True,
                 notes="O(n²) memory; scores a dataset relative to itself (cannot score new samples)."),
]}
DEFAULT = ["iforest", "knn", "lof", "cluster", "histogram", "pca", "mcd", "svm"]


def get(key: str) -> DetectorSpec:
    if key not in REGISTRY:
        raise KeyError(f"Unknown detector '{key}'. Available: {list(REGISTRY)}")
    return REGISTRY[key]


def fit_detector(Z: pd.DataFrame, key: str, contamination: float = 0.05, params: Optional[dict] = None,
                 random_state: int = 0):
    """(model, training scores (higher = more anomalous), 0/1 flags)."""
    model = get(key).factory(random_state, contamination, dict(params or {}))
    v = Z.to_numpy(float)
    model.fit(v)
    scores = np.asarray(model.decision_scores_, float)
    flags = np.asarray(model.labels_, int)
    return model, scores, flags


class TransductiveError(ValueError):
    pass


def score_new(model, Z_new: pd.DataFrame, key: Optional[str] = None):
    if key is not None and get(key).transductive:
        raise TransductiveError(f"{get(key).name} scores a dataset only relative to itself, so it cannot score "
                                "new samples against the training data. Use an inductive detector (e.g. "
                                "Isolation Forest, KNN, LOF, MCD) for new data.")
    v = Z_new.to_numpy(float)
    s = np.asarray(model.decision_function(v), float)
    return s, (s > model.threshold_).astype(int)


def label_metrics(scores, flags, reference) -> dict:
    ref = np.asarray(reference).astype(int)
    n_true = int(ref.sum())
    order = np.argsort(-scores)
    p_at_n = float(ref[order[:n_true]].mean()) if n_true else np.nan
    tp = int(((flags == 1) & (ref == 1)).sum())
    return {"roc_auc": M.roc_auc_score(ref, scores) if 0 < n_true < len(ref) else np.nan,
            "average_precision": M.average_precision_score(ref, scores) if n_true else np.nan,
            "precision_at_n": p_at_n,
            "precision_flagged": tp / max(int(flags.sum()), 1),
            "recall_flagged": tp / max(n_true, 1)}


ANOMALY_LABELS = {"n_flagged": "Flagged", "roc_auc": "ROC-AUC (vs reference)", "average_precision": "PR-AUC (AP)",
                  "precision_at_n": "Precision@n", "precision_flagged": "Precision (flagged)",
                  "recall_flagged": "Recall (flagged)", "fit_time_s": "Fit time (s)"}


@dataclass
class DetectorComparison:
    table: pd.DataFrame
    scores: pd.DataFrame          # columns = detector keys (higher = more anomalous)
    flags: pd.DataFrame
    agreement: pd.DataFrame       # Spearman between score vectors
    flag_jaccard: pd.DataFrame
    consensus: pd.Series          # mean percentile rank across detectors
    failures: dict


def compare_detectors(Z: pd.DataFrame, keys: list, contamination: float = 0.05, reference=None,
                      random_state: int = 0, progress=None) -> DetectorComparison:
    rows, S, F, fail = [], {}, {}, {}
    for i, k in enumerate(keys):
        spec = get(k)
        if progress:
            progress(i, len(keys), spec.name)
        t = time.time()
        try:
            _, s, f = fit_detector(Z, k, contamination, random_state=random_state)
        except Exception as e:  # noqa: BLE001
            fail[k] = f"{type(e).__name__}: {e}"
            continue
        S[k], F[k] = s, f
        row = {"key": k, "Model": spec.name, "family": spec.family, "n_flagged": int(f.sum()),
               "fit_time_s": time.time() - t}
        if reference is not None:
            row.update(label_metrics(s, f, reference))
        rows.append(row)
    tab = pd.DataFrame(rows)
    if reference is not None and len(tab):
        tab = tab.sort_values("average_precision", ascending=False, na_position="last").reset_index(drop=True)
    Sd, Fd = pd.DataFrame(S, index=Z.index), pd.DataFrame(F, index=Z.index)
    agree = Sd.rank().corr() if Sd.shape[1] > 1 else pd.DataFrame()
    jac = pd.DataFrame(index=Fd.columns, columns=Fd.columns, dtype=float)
    for a in Fd:
        for b in Fd:
            A, B = Fd[a] == 1, Fd[b] == 1
            jac.loc[a, b] = (A & B).sum() / max((A | B).sum(), 1)
    cons = Sd.rank(pct=True).mean(axis=1) if Sd.shape[1] else pd.Series(dtype=float)
    return DetectorComparison(tab, Sd, Fd, agree, jac, cons, fail)


def flag_stability(Z: pd.DataFrame, key: str, contamination: float = 0.05, params: Optional[dict] = None,
                   n_boot: int = 20, frac: float = 0.8, random_state: int = 0) -> dict:
    """Refit on subsamples; score *all* samples with each refit; report how often each sample is flagged."""
    rng = np.random.default_rng(random_state)
    _, s0, f0 = fit_detector(Z, key, contamination, params, random_state)
    counts = np.zeros(len(Z))
    jac = []
    seen = np.zeros(len(Z))
    trans = get(key).transductive
    for b in range(n_boot):
        ii = np.sort(rng.choice(len(Z), int(frac * len(Z)), replace=False))
        m = get(key).factory(random_state + b + 1, contamination, dict(params or {}))
        m.fit(Z.to_numpy(float)[ii])
        if trans:                            # flags exist only for the subsample members
            fl_sub = np.asarray(m.labels_, int)
            counts[ii] += fl_sub
            seen[ii] += 1
            A, B = f0[ii] == 1, fl_sub == 1
        else:
            _, fl = score_new(m, Z)
            counts += fl
            seen += 1
            A, B = f0 == 1, fl == 1
        jac.append((A & B).sum() / max((A | B).sum(), 1))
    freq = counts / np.maximum(seen, 1)
    return {"flag_frequency": pd.Series(freq, index=Z.index), "mean_jaccard": float(np.mean(jac)),
            "sd_jaccard": float(np.std(jac, ddof=1)), "n_boot": n_boot,
            "stable_flags": int(((f0 == 1) & (freq >= 0.8)).sum()), "n_flagged": int(f0.sum())}


def deviation_profile(X: pd.DataFrame, numeric: list, row_index) -> pd.DataFrame:
    """Robust z-score (median / MAD over all samples) of each numeric feature for one sample."""
    V = X[numeric].astype(float)
    med = V.median()
    mad = (V - med).abs().median() * 1.4826
    z = (V.loc[row_index] - med) / mad.replace(0, np.nan)
    out = pd.DataFrame({"feature": numeric, "value": V.loc[row_index].values, "median": med.values,
                        "robust_z": z.values})
    out["abs_z"] = out["robust_z"].abs()
    return out.sort_values("abs_z", ascending=False).reset_index(drop=True)
