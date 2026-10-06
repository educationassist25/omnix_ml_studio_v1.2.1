"""Model bundles: save / load with metadata and an integrity signature; new-sample prediction.

Security: a bundle is a zip of ``model.joblib`` + ``metadata.json`` + ``signature``
(HMAC-SHA256 over both). :func:`load_bundle` verifies the signature *before* unpickling and
refuses anything else, so arbitrary uploaded pickles are never executed. The key comes from
the ``OMNIX_MODEL_KEY`` environment variable (set it on the server to reload bundles across
sessions); without it, a per-process random key is used and bundles reload only in the same
running app.
"""
from __future__ import annotations

import hashlib
import hmac
import io
import json
import os
import platform
import secrets
import time
import zipfile
from importlib import metadata as md
from typing import Optional

import joblib
import numpy as np
import pandas as pd

_PROCESS_KEY = secrets.token_bytes(32)
BUNDLE_VERSION = 1


class BundleError(ValueError):
    pass


def _key() -> bytes:
    k = os.environ.get("OMNIX_MODEL_KEY")
    return k.encode() if k else _PROCESS_KEY


def key_is_persistent() -> bool:
    return bool(os.environ.get("OMNIX_MODEL_KEY"))


def dataset_fingerprint(X: pd.DataFrame, y=None) -> str:
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(X, index=False).values.tobytes())
    h.update("|".join(map(str, X.columns)).encode())
    if y is not None:
        h.update(np.asarray(y).tobytes())
    return h.hexdigest()[:20]


def software_versions() -> dict:
    out = {"python": platform.python_version()}
    for p in ("scikit-learn", "imbalanced-learn", "numpy", "pandas", "lightgbm", "xgboost", "catboost",
              "sktime", "pyod", "shap", "streamlit"):
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            pass
    return out


def training_profile(X: pd.DataFrame, numeric: list, categorical: list) -> dict:
    """Ranges / levels used for compatibility and out-of-distribution checks at prediction time."""
    return {"numeric": {c: [float(np.nanmin(X[c])), float(np.nanmax(X[c])), float(np.nanmedian(X[c]))]
                        for c in numeric if X[c].notna().any()},
            "categorical": {c: sorted(map(str, X[c].dropna().unique()))[:500] for c in categorical},
            "feature_order": list(X.columns)}


def make_bundle(model, meta: dict, ood_model=None) -> bytes:
    payload = io.BytesIO()
    joblib.dump({"model": model, "ood": ood_model}, payload, compress=3)
    model_bytes = payload.getvalue()
    meta = {**meta, "bundle_version": BUNDLE_VERSION, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "software": software_versions(), "model_sha256": hashlib.sha256(model_bytes).hexdigest()}
    meta_bytes = json.dumps(meta, indent=2, default=str).encode()
    sig = hmac.new(_key(), model_bytes + b"\0" + meta_bytes, hashlib.sha256).hexdigest()
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("model.joblib", model_bytes)
        z.writestr("metadata.json", meta_bytes)
        z.writestr("signature", sig)
    return out.getvalue()


def read_metadata(data: bytes) -> dict:
    """Metadata only — safe to call on any upload (no unpickling)."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            return json.loads(z.read("metadata.json"))
    except Exception as e:  # noqa: BLE001
        raise BundleError(f"Not an Omnix model bundle: {e}") from e


def load_bundle(data: bytes) -> tuple[object, Optional[object], dict]:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            names = set(z.namelist())
            if names != {"model.joblib", "metadata.json", "signature"}:
                raise BundleError("Unexpected bundle contents.")
            mb, meta_b, sig = z.read("model.joblib"), z.read("metadata.json"), z.read("signature").decode()
    except zipfile.BadZipFile as e:
        raise BundleError("Not an Omnix model bundle (not a zip file).") from e
    expect = hmac.new(_key(), mb + b"\0" + meta_b, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expect, sig):
        raise BundleError("Signature check failed: this bundle was not created by this server (or was modified). "
                          "It will not be loaded." + ("" if key_is_persistent() else
                                                      " Set OMNIX_MODEL_KEY on the server to reload bundles "
                                                      "after a restart."))
    obj = joblib.load(io.BytesIO(mb))
    return obj["model"], obj.get("ood"), json.loads(meta_b)


# --------------------------------------------------------------------------- prediction checks
def check_new_data(df: pd.DataFrame, profile: dict) -> dict:
    feats = profile["feature_order"]
    missing = [c for c in feats if c not in df.columns]
    extra = [c for c in df.columns if c not in feats]
    unseen = {}
    for c, levels in profile["categorical"].items():
        if c in df:
            new = sorted(set(df[c].dropna().astype(str)) - set(levels))
            if new:
                unseen[c] = new[:10]
    out_of_range = pd.Series(0, index=df.index, dtype=int)
    n_num = 0
    for c, (lo, hi, _) in profile["numeric"].items():
        if c in df:
            v = pd.to_numeric(df[c], errors="coerce")
            out_of_range += ((v < lo) | (v > hi)).astype(int)
            n_num += 1
    return {"missing": missing, "extra": extra, "unseen_levels": unseen,
            "frac_out_of_range": out_of_range / max(n_num, 1)}


def fit_ood_detector(Z: pd.DataFrame, random_state: int = 0):
    """Isolation forest on the model-input space of the training data (for OOD flags)."""
    from sklearn.ensemble import IsolationForest
    f = IsolationForest(n_estimators=200, contamination="auto", random_state=random_state).fit(Z.to_numpy(float))
    f.train_scores_ = -f.score_samples(Z.to_numpy(float))
    return f


def ood_scores(ood, Z: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(score, percentile vs training) — percentile > 99 means more unusual than 99% of training samples."""
    s = -ood.score_samples(Z.to_numpy(float))
    pct = np.searchsorted(np.sort(ood.train_scores_), s) / len(ood.train_scores_) * 100
    return s, pct


def prepare_features(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Columns in training order, categoricals coded as at training time; missing columns -> NaN."""
    cat = set(meta.get("categorical", []))
    cols = {}
    for c in meta["profile"]["feature_order"]:
        v = df[c] if c in df else pd.Series(np.nan, index=df.index)
        if c in cat:
            v = v.astype("object").where(v.notna(), np.nan).map(lambda x: x if pd.isna(x) else str(x))
        else:
            v = pd.to_numeric(v, errors="coerce")
        cols[c] = v
    return pd.DataFrame(cols, index=df.index)      # built at once: fast with 10,000+ columns


def predict_new(model, meta: dict, df: pd.DataFrame, ood=None, id_col: Optional[str] = None,
                ood_pct: float = 99.0, range_frac: float = 0.2) -> pd.DataFrame:
    """Predictions for new samples, with compatibility and out-of-distribution flags.

    A flag does not mean the prediction is wrong; it means the sample is unlike the
    training data, so the model's validated performance may not apply to it.
    """
    from .explainability import model_input
    X = prepare_features(df, meta)
    chk = check_new_data(df, meta["profile"])
    out = pd.DataFrame(index=df.index)
    if id_col and id_col in df:
        out[id_col] = df[id_col].values
    task = meta["task"]
    if task == "regression":
        out["prediction"] = model.predict(X)
    else:
        classes = meta["classes"]
        if hasattr(model, "predict_proba"):
            try:
                P = model.predict_proba(X)
                out["predicted_class"] = [classes[i] for i in np.argmax(P, 1)]
                for j, c in enumerate(classes):
                    out[f"P({c})"] = P[:, j]
            except AttributeError:
                out["predicted_class"] = [classes[int(i)] for i in model.predict(X)]
        else:
            out["predicted_class"] = [classes[int(i)] for i in model.predict(X)]
    out["out_of_range_fraction"] = chk["frac_out_of_range"].values
    flags = out["out_of_range_fraction"] > range_frac
    if ood is not None:
        _, pct = ood_scores(ood, model_input(model, X))
        out["ood_percentile"] = np.round(pct, 1)
        flags = flags | (out["ood_percentile"] > ood_pct)
    out["flag"] = np.where(flags, "unlike training data", "")
    out.attrs["check"] = chk
    return out
