"""Synthetic datasets with known ground truth, for tests and examples.

All generators return a DataFrame and are deterministic for a given ``seed``.
Feature names are ``f0001``... so they look like anonymised omics features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _features(rng, n, p, names=None):
    X = rng.normal(size=(n, p))
    return pd.DataFrame(X, columns=names or [f"f{i:04d}" for i in range(1, p + 1)])


def binary(n=300, p=20, n_informative=5, seed=0, missing=0.02, with_categorical=True,
           effect=1.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = _features(rng, n, p)
    w = np.zeros(p); w[:n_informative] = effect * rng.choice([-1, 1], n_informative)
    lp = X.values @ w
    if with_categorical:
        sex = rng.choice(["F", "M"], n)
        stage = rng.choice(["I", "II", "III"], n, p=[.5, .3, .2])
        lp = lp + 0.8 * (stage == "III") + 0.3 * (sex == "M")
    y = rng.binomial(1, 1 / (1 + np.exp(-(lp - np.median(lp)))))
    df = X.copy()
    if with_categorical:
        df.insert(0, "stage", stage); df.insert(0, "sex", sex)
    if missing:
        m = rng.random(df[X.columns].shape) < missing
        df[X.columns] = df[X.columns].mask(m)
    df.insert(0, "sample_id", [f"S{i:04d}" for i in range(n)])
    df["outcome"] = np.where(y == 1, "case", "control")
    return df


def multiclass(n=360, p=20, n_classes=3, seed=1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = _features(rng, n, p)
    W = rng.normal(0, 1.2, (p, n_classes)); W[6:] = 0
    logits = X.values @ W
    pr = np.exp(logits - logits.max(1, keepdims=True)); pr /= pr.sum(1, keepdims=True)
    y = np.array([rng.choice(n_classes, p=r) for r in pr])
    df = X.copy()
    df.insert(0, "sample_id", [f"S{i:04d}" for i in range(n)])
    df["subtype"] = np.array(["luminal", "basal", "neuronal"][:n_classes] +
                             [f"c{k}" for k in range(3, n_classes)])[y]
    return df


def regression(n=300, p=20, n_informative=5, seed=2, noise=1.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = _features(rng, n, p)
    w = np.zeros(p); w[:n_informative] = rng.uniform(0.5, 1.5, n_informative)
    y = X.values @ w + 0.5 * X.values[:, 0] ** 2 + rng.normal(0, noise, n)
    df = X.copy()
    df.insert(0, "sample_id", [f"S{i:04d}" for i in range(n)])
    df["response"] = y + 50
    return df


def omics_hdlss(n=80, p=2000, n_informative=10, seed=3, effect=0.9, null=False) -> pd.DataFrame:
    """High-dimensional, low-sample-size, positive log-normal intensities."""
    rng = np.random.default_rng(seed)
    y = rng.binomial(1, 0.5, n)
    Z = rng.normal(size=(n, p))
    if not null:
        Z[:, :n_informative] += effect * (2 * y[:, None] - 1) * rng.choice([-1, 1], n_informative)
    X = np.exp(10 + 0.8 * Z)          # positive, right-skewed like MS intensities
    zero = rng.random(X.shape) < 0.01
    X[zero] = 0.0                     # non-detects (log transform needs explicit offset)
    df = pd.DataFrame(X, columns=[f"m{i:05d}" for i in range(1, p + 1)])
    df.insert(0, "sample_id", [f"S{i:03d}" for i in range(n)])
    df["group"] = np.where(y == 1, "tumor", "normal")
    return df


def imbalanced(n=400, p=15, minority=0.08, seed=4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < minority).astype(int)
    X = _features(rng, n, p)
    X.iloc[:, :3] += 1.0 * y[:, None]
    df = X.copy()
    df.insert(0, "sample_id", [f"S{i:04d}" for i in range(n)])
    df["event"] = np.where(y == 1, "yes", "no")
    return df


def grouped_repeated(n_patients=60, visits=4, p=30, seed=5) -> pd.DataFrame:
    """Repeated measurements with a strong patient fingerprint and NO real signal.

    Outcome is assigned per patient at random, so the honest AUC is 0.5. Each patient
    has a persistent feature 'fingerprint'; if visits of the same patient land in both
    training and test folds, a flexible model memorises the fingerprint and AUC is
    falsely high. GroupKFold must bring it back to ~0.5.
    """
    rng = np.random.default_rng(seed)
    fp = rng.normal(0, 1.5, (n_patients, p))
    yp = rng.binomial(1, 0.5, n_patients)
    rows = []
    for i in range(n_patients):
        for _v in range(visits):
            rows.append(np.r_[fp[i] + rng.normal(0, 0.3, p)])
    df = pd.DataFrame(rows, columns=[f"f{j:04d}" for j in range(1, p + 1)])
    pid = np.repeat([f"P{i:03d}" for i in range(n_patients)], visits)
    df.insert(0, "visit", np.tile(np.arange(visits), n_patients))
    df.insert(0, "patient_id", pid)
    df.insert(0, "sample_id", [f"{a}_V{b}" for a, b in zip(pid, df["visit"])])
    df["response"] = np.where(np.repeat(yp, visits) == 1, "responder", "non_responder")
    return df


def batch_confounded(n=200, p=20, seed=6) -> pd.DataFrame:
    """Batch strongly aligned with outcome; batch shifts all features."""
    rng = np.random.default_rng(seed)
    y = rng.binomial(1, 0.5, n)
    batch = np.where(rng.random(n) < 0.9, y, 1 - y)
    X = _features(rng, n, p) + 1.0 * batch[:, None]
    df = X.copy()
    df.insert(0, "batch", np.where(batch == 1, "B2", "B1"))
    df.insert(0, "sample_id", [f"S{i:04d}" for i in range(n)])
    df["outcome"] = np.where(y == 1, "case", "control")
    return df


def external_pair(n_dev=300, n_ext=150, p=20, seed=7, shift=0.5):
    """Development and external cohort from the same model; external has a mean shift."""
    both = binary(n_dev + n_ext, p, seed=seed, missing=0.0, with_categorical=False)
    dev = both.iloc[:n_dev].reset_index(drop=True)
    ext = both.iloc[n_dev:].reset_index(drop=True)
    ext["sample_id"] = [f"E{i:04d}" for i in range(n_ext)]
    cols = [c for c in ext.columns if c.startswith("f")]
    ext[cols[5:10]] = ext[cols[5:10]] + shift   # shift non-informative features
    return dev, ext


def with_leak(df: pd.DataFrame, target: str, name="leaky_marker", seed=0) -> pd.DataFrame:
    """Add a feature that is a noisy copy of the outcome (deliberate target leakage)."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    yv = pd.factorize(out[target], sort=True)[0]
    out[name] = yv + rng.normal(0, 0.01, len(out))
    return out


def clusters(n=450, p=12, k=3, seed=8, separation=3.0, noise_features=6) -> pd.DataFrame:
    """k Gaussian clusters in the first p - noise_features dims; reference label kept for evaluation."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, separation, (k, p - noise_features))
    lab = rng.integers(0, k, n)
    X = np.c_[centers[lab] + rng.normal(0, 1, (n, p - noise_features)), rng.normal(0, 1, (n, noise_features))]
    df = pd.DataFrame(X, columns=[f"f{i:04d}" for i in range(1, p + 1)])
    df.insert(0, "sample_id", [f"S{i:04d}" for i in range(n)])
    df["true_subtype"] = np.array(["A", "B", "C", "D", "E", "F"])[lab]
    return df


def anomalies(n=500, p=10, frac=0.05, seed=9, shift=(4, 6)) -> pd.DataFrame:
    """Correlated inliers + a known fraction of anomalies (shifted in a random subset of features)."""
    rng = np.random.default_rng(seed)
    C = 0.5 * np.ones((p, p)) + 0.5 * np.eye(p)
    X = rng.multivariate_normal(np.zeros(p), C, n)
    is_out = rng.random(n) < frac
    for i in np.where(is_out)[0]:
        j = rng.choice(p, 3, replace=False)
        X[i, j] += rng.choice([-1, 1], 3) * rng.uniform(shift[0], shift[1], 3)
    df = pd.DataFrame(X, columns=[f"f{i:04d}" for i in range(1, p + 1)])
    df.insert(0, "sample_id", [f"S{i:04d}" for i in range(n)])
    df["is_anomaly"] = is_out.astype(int)
    return df


def monthly_series(n=144, seed=10, trend=0.8, season_amp=12.0, noise=3.0, start="2012-01-01") -> pd.DataFrame:
    """Monthly series with linear trend + multiplicative-looking yearly seasonality."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    level = 100 + trend * t
    season = season_amp * np.sin(2 * np.pi * t / 12) * (1 + t / n)
    y = level + season + rng.normal(0, noise, n)
    return pd.DataFrame({"date": pd.date_range(start, periods=n, freq="MS"), "value": y.round(2)})
