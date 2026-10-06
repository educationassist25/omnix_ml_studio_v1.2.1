"""Model interpretation for fitted pipelines.

* SHAP (tree / linear / permutation explainers chosen by model type)
* Permutation importance (on raw features, full pipeline, chosen evaluation set)
* Model-specific importance (|coef| or impurity importance)
* Partial dependence and individual conditional expectation (PDP / ICE)

Encoded columns (one-hot ``col_level``) are mapped back to original features by
summing their contributions. Explanations describe the *model*, not the biology:
importance does not imply causality or statistical significance.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .feature_selection import to_original
from .metrics import primary_metric, scorers

TREE = ("RandomForest", "ExtraTrees", "GradientBoosting", "DecisionTree", "LGBM", "XGB", "CatBoost")
LINEAR = ("LogisticRegression", "Ridge", "Lasso", "ElasticNet", "LinearRegression",
          "LinearDiscriminantAnalysis")
MAX_PERMUTATION_SHAP_FEATURES = 300

CAVEAT = ("Importance describes how this fitted model uses its inputs. It does not establish "
          "causality, biological mechanism or statistical significance, and correlated features "
          "can share or swap importance.")


def model_input(pipe, X: pd.DataFrame) -> pd.DataFrame:
    """Apply every pipeline step except the model (samplers are skipped, as at predict time)."""
    Z = X
    for _, step in pipe.steps[:-1]:
        if hasattr(step, "fit_resample"):
            continue
        Z = step.transform(Z)
    return pd.DataFrame(Z)


def final_estimator(pipe):
    return pipe.steps[-1][1]


def explainer_kind(model) -> str:
    n = type(model).__name__
    if n.startswith(TREE):
        return "tree"
    if n.startswith(LINEAR):
        return "linear"
    return "permutation"


@dataclass
class ShapResult:
    values: np.ndarray            # (n, p, n_outputs) on encoded model features
    base_values: np.ndarray       # (n, n_outputs)
    data: pd.DataFrame            # model-input values for the explained rows (encoded)
    feature_names: list
    output_names: list
    kind: str
    output_space: str
    row_index: pd.Index           # original row index of explained samples

    def for_output(self, k: int) -> np.ndarray:
        return self.values[:, :, k]


def _output_space(model, kind, task):
    n = type(model).__name__
    if task == "regression":
        return "prediction"
    if kind == "linear":
        return "log-odds" if n.startswith("LogisticRegression") else "decision function"
    if n.startswith(("GradientBoosting", "LGBM", "XGB", "CatBoost")):
        return "log-odds"
    return "probability"


def explain_shap(pipe, X_explain: pd.DataFrame, X_background: pd.DataFrame, task: str,
                 classes: Optional[list] = None, max_explain: int = 200, max_background: int = 100,
                 random_state: int = 0) -> ShapResult:
    import shap
    rng = np.random.default_rng(random_state)
    if len(X_explain) > max_explain:
        X_explain = X_explain.iloc[np.sort(rng.choice(len(X_explain), max_explain, replace=False))]
    if len(X_background) > max_background:
        X_background = X_background.iloc[np.sort(rng.choice(len(X_background), max_background, replace=False))]
    model = final_estimator(pipe)
    Z, B = model_input(pipe, X_explain), model_input(pipe, X_background)
    names = list(Z.columns)
    kind = explainer_kind(model)
    space = _output_space(model, kind, task)
    if kind == "tree":
        ex = shap.TreeExplainer(model)
        sv = ex(Z, check_additivity=False)
    elif kind == "linear":
        ex = shap.LinearExplainer(model, B)
        sv = ex(Z)
    else:
        if Z.shape[1] > MAX_PERMUTATION_SHAP_FEATURES:
            raise ValueError(f"Model-agnostic SHAP is limited to {MAX_PERMUTATION_SHAP_FEATURES} model "
                             f"features (this model receives {Z.shape[1]}). Use permutation importance, "
                             "or add feature selection.")
        f = model.predict_proba if (task != "regression" and hasattr(model, "predict_proba")) else model.predict
        ex = shap.PermutationExplainer(f, shap.maskers.Independent(B, max_samples=len(B)), seed=random_state)
        sv = ex(Z, max_evals=2 * Z.shape[1] + 1, silent=True)
        if task != "regression":
            space = "probability"
    vals = np.asarray(sv.values, float)
    base = np.asarray(sv.base_values, float)
    if vals.ndim == 2:
        vals = vals[:, :, None]
    n, _, k = vals.shape
    if base.ndim == 0:
        base = np.full((n, k), float(base))
    elif base.ndim == 1:
        base = base[:, None] if base.size == n else np.tile(base[None, :], (n, 1))
    base = np.broadcast_to(base, (n, k)).copy()
    if task == "regression":
        outs = ["prediction"]
    elif k == 1:                                   # binary models with a single output
        outs = [str(classes[-1]) if classes else "positive"]
    else:
        outs = [str(c) for c in classes] if classes and len(classes) == k else [f"class {i}" for i in range(k)]
    return ShapResult(vals, base, Z, names, outs, kind, space, X_explain.index)


def positive_output_index(res: ShapResult) -> int:
    """Index of the output to show by default (positive class for binary)."""
    return len(res.output_names) - 1


def aggregate_to_original(values: np.ndarray, names: list, categorical: list) -> tuple[np.ndarray, list]:
    """Sum SHAP columns of one-hot levels back to original features. values: (n, p)."""
    orig = [to_original(c, categorical) for c in names]
    uniq = list(dict.fromkeys(orig))
    out = np.zeros((values.shape[0], len(uniq)))
    pos = {u: i for i, u in enumerate(uniq)}
    for j, o in enumerate(orig):
        out[:, pos[o]] += values[:, j]
    return out, uniq


def shap_importance(res: ShapResult, output: int, categorical: list) -> pd.DataFrame:
    v, names = aggregate_to_original(res.for_output(output), res.feature_names, categorical)
    d = pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(v).mean(0),
                      "mean_shap": v.mean(0)})
    return d.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def permutation_importances(pipe, X: pd.DataFrame, y, task: str, n_classes: int = 2, n_repeats: int = 10,
                            random_state: int = 0, metric: Optional[str] = None,
                            features: Optional[list] = None, categorical: Optional[list] = None,
                            progress=None, should_stop=None) -> pd.DataFrame:
    """Drop in ``metric`` when each *original* feature is shuffled (full pipeline).

    ``features`` restricts the computation (e.g. to features the model can use); every
    other feature has importance exactly zero by construction and is listed as such.
    """
    from sklearn.metrics import check_scoring
    metric = metric or primary_metric(task)
    scorer = scorers(task, n_classes, y)[metric]
    scorer = check_scoring(pipe, scoring=scorer)
    feats = list(features) if features is not None else list(X.columns)
    rng = np.random.default_rng(random_state)
    fast = _permutation_fast(pipe, X, y, scorer, feats, n_repeats, rng, categorical, progress, should_stop) \
        if categorical is not None else None
    if fast is not None:
        base, rows = fast
        return _perm_frame(rows, X, feats, base, metric)
    base = scorer(pipe, X, y)
    rows = []
    for f in feats:
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[f] = rng.permutation(Xp[f].to_numpy())
            drops.append(base - scorer(pipe, Xp, y))
        rows.append({"feature": f, "importance_mean": float(np.mean(drops)),
                     "importance_sd": float(np.std(drops, ddof=1)) if n_repeats > 1 else np.nan})
    return _perm_frame(rows, X, feats, base, metric)


def _columnwise(pipe) -> bool:
    """True if every preprocessing step acts on each column separately (so shuffling commutes with it)."""
    from sklearn.impute import KNNImputer
    try:
        from sklearn.impute import IterativeImputer
    except ImportError:            # experimental import not enabled
        IterativeImputer = ()      # noqa: N806
    from sklearn.decomposition import PCA
    for _, step in pipe.steps[:-1]:
        for obj in _walk(step):
            if isinstance(obj, (KNNImputer, PCA) + ((IterativeImputer,) if IterativeImputer else ())):
                return False
    return True


def _walk(obj):
    yield obj
    for attr in ("steps", "transformers", "transformers_"):
        for item in getattr(obj, attr, []) or []:
            sub = item[1] if isinstance(item, tuple) else item
            if hasattr(sub, "get_params"):
                yield from _walk(sub)


def _permutation_fast(pipe, X, y, scorer, feats, n_repeats, rng, categorical, progress=None, should_stop=None):
    """Transform once, then shuffle each original feature's model-input columns (same row order
    for all one-hot levels). Identical to shuffling the raw column and re-running the pipeline
    when preprocessing is column-wise; ~100× faster with thousands of features."""
    if not _columnwise(pipe):
        return None
    try:
        Z = model_input(pipe, X).to_numpy(float)
        names = model_input_names(pipe)
    except Exception:  # noqa: BLE001
        return None
    if Z.shape[1] != len(names):
        return None
    model = final_estimator(pipe)
    msc = scorer                      # a scikit-learn scorer works on the bare model and its input
    # (thread pools per predict call cost more than they save on small evaluation sets: measured)
    return _permute_columns(model, msc, Z, names, y, feats, n_repeats, rng, categorical, progress, should_stop)


def _permute_columns(model, msc, Z, names, y, feats, n_repeats, rng, categorical, progress=None, should_stop=None):
    orig = np.array([to_original(n, categorical) for n in names])
    base = msc(model, Z, y)
    rows = []
    batch = n_repeats * Z.size * 8 < 256 * 2 ** 20          # all shuffles of one feature in one predict call
    for i, f in enumerate(feats):
        if should_stop and should_stop():
            break
        if progress and i % 20 == 0:
            progress(i, len(feats), f)
        cols = np.where(orig == f)[0]
        perms = [rng.permutation(len(Z)) for _ in range(n_repeats)]
        if not len(cols):
            drops = [0.0] * n_repeats
        else:
            blocks = []
            for perm in perms:
                Zp = Z.copy()
                Zp[:, cols] = Z[perm][:, cols]
                blocks.append(Zp)
            drops = None
            if batch:
                try:
                    rep = _Replay(model, blocks)
                    drops = [base - msc(rep, b, y) for b in blocks]
                except Exception:  # noqa: BLE001 - fall back to one call per shuffle
                    drops, batch = None, False
            if drops is None:
                drops = [base - msc(model, b, y) for b in blocks]
        rows.append({"feature": f, "importance_mean": float(np.mean(drops)),
                     "importance_sd": float(np.std(drops, ddof=1)) if n_repeats > 1 else np.nan})
    return base, rows


class _Replay:
    """Stands in for a fitted model: answers predict / predict_proba / decision_function for each
    of several equal-sized blocks from ONE call on their concatenation (per-call overhead of tree
    ensembles dominates on small evaluation sets)."""

    def __init__(self, model, blocks):
        self._model, self._blocks, self._out = model, blocks, {}
        for a in ("classes_", "_estimator_type", "n_features_in_"):
            if hasattr(model, a):
                setattr(self, a, getattr(model, a))
        for m in ("predict", "predict_proba", "decision_function", "predict_log_proba"):
            if hasattr(model, m):
                setattr(self, m, (lambda X, _m=m: self._get(_m, X)))

    def _get(self, meth, X):
        if meth not in self._out:
            big = np.vstack(self._blocks)
            o = np.asarray(getattr(self._model, meth)(big))
            n = len(self._blocks[0])
            self._out[meth] = {id(b): o[i * n:(i + 1) * n] for i, b in enumerate(self._blocks)}
        return self._out[meth][id(X)]


def _perm_frame(rows, X, feats, base, metric):
    out = pd.DataFrame(rows)
    rest = [c for c in X.columns if c not in set(feats)]
    if rest:
        out = pd.concat([out, pd.DataFrame({"feature": rest, "importance_mean": 0.0, "importance_sd": 0.0,
                                            "note": "not used by the model"})], ignore_index=True)
    out.attrs["baseline_score"] = float(base)
    out.attrs["metric"] = metric
    return out.sort_values("importance_mean", ascending=False).reset_index(drop=True)


def model_features_original(pipe, categorical: list) -> list:
    """Original feature names that actually reach the model after filtering/selection."""
    Z_names = model_input_names(pipe)
    return list(dict.fromkeys(to_original(c, categorical) for c in Z_names))


def model_input_names(pipe) -> list:
    steps = dict(pipe.steps)
    names = np.asarray(steps["columns"].get_feature_names_out())
    for k in ("select", "fs"):
        if k in steps:
            names = names[steps[k].get_support()]
    return list(names)


def model_specific_importance(pipe, categorical: list) -> Optional[pd.DataFrame]:
    """|coef| (on the model's input scale) or impurity-based importance."""
    m = final_estimator(pipe)
    names = model_input_names(pipe)
    if hasattr(m, "feature_importances_"):
        imp, kind = np.asarray(m.feature_importances_, float), "impurity / split-gain importance"
    elif hasattr(m, "coef_"):
        imp, kind = np.abs(np.atleast_2d(m.coef_)).max(0), "|coefficient| (max over classes)"
    else:
        return None
    if len(imp) != len(names):
        return None
    d = pd.DataFrame({"encoded_feature": names, "feature": [to_original(c, categorical) for c in names],
                      "importance": imp})
    d.attrs["kind"] = kind
    return d.sort_values("importance", ascending=False).reset_index(drop=True)


def pdp_ice(pipe, X: pd.DataFrame, feature: str, task: str, categorical: list, output: int = -1,
            grid_resolution: int = 20, n_ice: int = 50, random_state: int = 0) -> dict:
    """Partial dependence (average) and ICE curves (subset of rows) for one original feature."""
    from sklearn.inspection import partial_dependence
    is_cat = feature in categorical
    kw = dict(features=[feature], kind="both", grid_resolution=grid_resolution)
    if is_cat:
        kw["categorical_features"] = [feature]
    if task != "regression":
        kw["response_method"] = "predict_proba"
    rng = np.random.default_rng(random_state)
    Xs = X if len(X) <= 400 else X.iloc[rng.choice(len(X), 400, replace=False)]
    r = partial_dependence(pipe, Xs, **kw)
    avg, ind = np.asarray(r["average"]), np.asarray(r["individual"])
    k = avg.shape[0]
    o = output if output >= 0 else k - 1
    o = min(o, k - 1)
    pick = rng.choice(ind.shape[1], min(n_ice, ind.shape[1]), replace=False)
    grid = r["grid_values"][0] if "grid_values" in r else r["values"][0]
    return {"grid": np.asarray(grid), "average": avg[o], "ice": ind[o][pick], "categorical": is_cat,
            "response": "prediction" if task == "regression" else "predicted probability",
            "n_rows": len(Xs)}
