import time

import numpy as np
import pytest
from conftest import design_for

from ml_studio.core import synthetic
from ml_studio.core.config import PreprocessConfig, ValidationConfig
from ml_studio.core.explainability import (aggregate_to_original, explain_shap, model_features_original,
                                           model_input, model_specific_importance, pdp_ice,
                                           permutation_importances, shap_importance)
from ml_studio.core.preprocessing import prepare_xy
from ml_studio.core.splitting import holdout_split
from ml_studio.core.validation import fit_final

INFORMATIVE = {"f0001", "f0002", "f0003", "f0004", "f0005"}


@pytest.fixture(scope="module")
def bin_setup():
    df = synthetic.binary(n=400, p=20, seed=12, effect=1.3)
    d = prepare_xy(df, design_for(df, "outcome", "binary"))
    sp = holdout_split(d.y, "binary", 0.25, None, 0)
    return d, sp


def _fit(d, sp, key, task="binary", **cfg):
    return fit_final(d, sp.train_idx, task, key, PreprocessConfig(**cfg), ValidationConfig())


@pytest.mark.parametrize("key,space", [("lr", "log-odds"), ("rf", "probability"),
                                       ("lightgbm", "log-odds"), ("gbc", "log-odds")])
def test_shap_additivity_and_space(bin_setup, key, space):
    d, sp = bin_setup
    pipe = _fit(d, sp, key)
    Xe, Xb = d.X.iloc[sp.test_idx], d.X.iloc[sp.train_idx]
    r = explain_shap(pipe, Xe, Xb, "binary", d.classes, max_explain=60)
    assert r.output_space == space
    m = pipe.steps[-1][1]
    Z = model_input(pipe, Xe.loc[r.row_index])
    o = len(r.output_names) - 1
    recon = r.values[:, :, o].sum(1) + r.base_values[:, o]
    if space == "probability":
        target = m.predict_proba(Z)[:, 1]
    else:
        target = m.decision_function(Z) if hasattr(m, "decision_function") else m.predict(Z, raw_score=True)
        target = np.ravel(target)
    assert np.allclose(recon, target, atol=1e-4), np.abs(recon - target).max()


@pytest.mark.parametrize("key", ["lr", "rf", "lightgbm", "rbfsvm", "knn"])
def test_shap_recovers_informative_features(bin_setup, key):
    d, sp = bin_setup
    pipe = _fit(d, sp, key)
    r = explain_shap(pipe, d.X.iloc[sp.test_idx], d.X.iloc[sp.train_idx], "binary", d.classes,
                     max_explain=40 if key in ("rbfsvm", "knn") else 100, max_background=50)
    imp = shap_importance(r, len(r.output_names) - 1, d.categorical)
    top = set(imp["feature"].head(6))
    assert len(top & INFORMATIVE) >= 4, (key, imp.head(8))
    assert "stage" in set(imp["feature"]) and not any(f.startswith("stage_") for f in imp["feature"])


def test_aggregation_sums_one_hot_levels():
    v = np.array([[1.0, 2.0, 3.0, 4.0]])
    out, names = aggregate_to_original(v, ["age", "stage_I", "stage_II", "sex_M"], ["stage", "sex"])
    assert names == ["age", "stage", "sex"] and out.tolist() == [[1.0, 5.0, 4.0]]


def test_multiclass_and_regression_shap(multiclass_df, regression_df):
    d = prepare_xy(multiclass_df, design_for(multiclass_df, "subtype", "multiclass"))
    sp = holdout_split(d.y, "multiclass", 0.2, None, 0)
    for key in ("lr", "rf"):
        r = explain_shap(_fit(d, sp, key, "multiclass"), d.X.iloc[sp.test_idx], d.X.iloc[sp.train_idx],
                         "multiclass", d.classes, max_explain=40)
        assert r.values.shape[2] == 3 and r.output_names == d.classes
    d = prepare_xy(regression_df, design_for(regression_df, "response", "regression"))
    sp = holdout_split(d.y, "regression", 0.2, None, 0)
    pipe = _fit(d, sp, "rf", "regression")
    r = explain_shap(pipe, d.X.iloc[sp.test_idx], d.X.iloc[sp.train_idx], "regression", None, max_explain=40)
    Z = model_input(pipe, d.X.loc[r.row_index])
    assert np.allclose(r.values[:, :, 0].sum(1) + r.base_values[:, 0], pipe.steps[-1][1].predict(Z), atol=1e-6)
    assert len(set(shap_importance(r, 0, []).head(6)["feature"]) & INFORMATIVE) >= 4


def test_permutation_importance(bin_setup):
    d, sp = bin_setup
    pipe = _fit(d, sp, "lr")
    pi = permutation_importances(pipe, d.X.iloc[sp.test_idx], d.y[sp.test_idx], "binary", n_repeats=5)
    assert len(set(pi["feature"].head(6)) & INFORMATIVE) >= 4
    assert pi.attrs["metric"] == "roc_auc" and 0.7 < pi.attrs["baseline_score"] <= 1
    noise = pi[pi["feature"].isin(["f0015", "f0016", "f0017"])]["importance_mean"]
    assert (noise.abs() < 0.05).all()


def test_permutation_restricted_to_used_features(bin_setup):
    d, sp = bin_setup
    pipe = _fit(d, sp, "lr", fs_method="univariate", fs_k=5)
    used = model_features_original(pipe, d.categorical)
    assert len(used) == 5
    pi = permutation_importances(pipe, d.X.iloc[sp.test_idx], d.y[sp.test_idx], "binary", n_repeats=3,
                                 features=used)
    zero = pi[pi.get("note").notna()] if "note" in pi else pi.iloc[0:0]
    assert len(zero) == len(d.X.columns) - 5 and (zero["importance_mean"] == 0).all()


def test_model_specific_importance(bin_setup):
    d, sp = bin_setup
    for key in ("lr", "rf", "lightgbm"):
        mi = model_specific_importance(_fit(d, sp, key), d.categorical)
        assert mi is not None and len(set(mi["feature"].head(6)) & INFORMATIVE) >= 4, key
    assert model_specific_importance(_fit(d, sp, "knn"), d.categorical) is None


def test_pdp_ice_direction(bin_setup):
    d, sp = bin_setup
    pipe = _fit(d, sp, "lr")
    coef_sign = {}
    names = list(model_input(pipe, d.X.head(2)).columns)
    for f in ("f0001", "f0002"):
        coef_sign[f] = np.sign(pipe.steps[-1][1].coef_[0][names.index(f)])
        r = pdp_ice(pipe, d.X.iloc[sp.train_idx], f, "binary", d.categorical, grid_resolution=15, n_ice=20)
        slope = np.sign(r["average"][-1] - r["average"][0])
        assert slope == coef_sign[f]
        assert r["ice"].shape == (20, len(r["grid"]))
    r = pdp_ice(pipe, d.X.iloc[sp.train_idx], "stage", "binary", d.categorical)
    assert r["categorical"] and len(r["grid"]) == 3


def test_hdlss_explain_performance():
    df = synthetic.omics_hdlss(n=80, p=2000)
    d = prepare_xy(df, design_for(df, "group", "binary"))
    sp = holdout_split(d.y, "binary", 0.2, None, 0)
    t = time.time()
    pipe = fit_final(d, sp.train_idx, "binary", "lightgbm",
                     PreprocessConfig(transform="log2", log_offset=1.0, fs_method="l1", fs_k=30), ValidationConfig())
    r = explain_shap(pipe, d.X.iloc[sp.test_idx], d.X.iloc[sp.train_idx], "binary", d.classes)
    used = model_features_original(pipe, [])
    pi = permutation_importances(pipe, d.X.iloc[sp.test_idx], d.y[sp.test_idx], "binary", n_repeats=3, features=used)
    dt = time.time() - t
    assert dt < 90, dt
    assert r.values.shape[1] == 30 and len(pi) == 2000
    informative = {f"m{i:05d}" for i in range(1, 11)}
    assert len(set(shap_importance(r, 0, []).head(10)["feature"]) & informative) >= 6


def test_shap_refuses_agnostic_on_very_wide_models():
    df = synthetic.omics_hdlss(n=60, p=400)
    d = prepare_xy(df, design_for(df, "group", "binary"))
    pipe = fit_final(d, np.arange(60), "binary", "knn", PreprocessConfig(transform="log2", log_offset=1.0),
                     ValidationConfig())
    with pytest.raises(ValueError, match="permutation importance"):
        explain_shap(pipe, d.X.head(10), d.X, "binary", d.classes)
