import numpy as np
import pytest
from conftest import design_for

from ml_studio.core import synthetic
from ml_studio.core.config import PreprocessConfig, ValidationConfig
from ml_studio.core.optimization import tune
from ml_studio.core.preprocessing import prepare_xy
from ml_studio.core.validation import (HoldoutLedger, bootstrap_ci, evaluate_holdout,
                                       external_validate, fit_final, nested_cv)
from ml_studio.core.splitting import holdout_split


@pytest.fixture(scope="module")
def bin_data():
    df = synthetic.binary(n=240)
    return prepare_xy(df, design_for(df, "outcome", "binary", sample_id="sample_id"))


def test_nested_cv_runs_and_reports(bin_data):
    r = nested_cv(bin_data, np.arange(len(bin_data.y)), "binary", ["lr", "rf"], PreprocessConfig(),
                  ValidationConfig(n_folds=4, inner_folds=3), n_iter=4)
    assert len(r.outer) == 4
    assert set(r.selection_frequency.index) <= {"lr", "rf"}
    s = r.summary.set_index("metric")
    assert 0.7 < s.loc["roc_auc", "mean"] <= 1.0
    assert all(isinstance(p, dict) for p in r.outer["params"])


def test_nested_cv_reproducible(bin_data):
    kw = dict(prep_cfg=PreprocessConfig(), val_cfg=ValidationConfig(n_folds=3, inner_folds=3), n_iter=3)
    a = nested_cv(bin_data, np.arange(240), "binary", ["lr"], **kw)
    b = nested_cv(bin_data, np.arange(240), "binary", ["lr"], **kw)
    assert np.allclose(a.outer["outer_roc_auc"], b.outer["outer_roc_auc"])


def test_nested_cv_on_null_hdlss_is_not_optimistic():
    df = synthetic.omics_hdlss(n=80, p=1000, null=True, seed=21)
    d = prepare_xy(df, design_for(df, "group", "binary"))
    r = nested_cv(d, np.arange(80), "binary", ["lr", "lr_l1"],
                  PreprocessConfig(transform="log2", log_offset=1.0, univariate_k=20),
                  ValidationConfig(n_folds=4, inner_folds=3), n_iter=3)
    assert 0.25 < r.summary.set_index("metric").loc["roc_auc", "mean"] < 0.72
    assert len(r.selected_features) == 4 and all(len(f) == 20 for f in r.selected_features)


def test_nested_grouped(grouped_df):
    d = prepare_xy(grouped_df, design_for(grouped_df, "response", "binary",
                                          exclude=("sample_id", "patient_id", "visit"),
                                          group="patient_id"))
    r = nested_cv(d, np.arange(len(d.y)), "binary", ["knn"], PreprocessConfig(),
                  ValidationConfig(cv_strategy="stratified_group_kfold", n_folds=4, inner_folds=3), n_iter=2)
    g = d.groups
    for tr, te in r.outer_splits:
        assert not set(g[tr]) & set(g[te])
    assert r.summary.set_index("metric").loc["roc_auc", "mean"] < 0.7


def test_multiclass_and_regression_nested(multiclass_df, regression_df):
    d = prepare_xy(multiclass_df, design_for(multiclass_df, "subtype", "multiclass"))
    r = nested_cv(d, np.arange(len(d.y)), "multiclass", ["lr"], PreprocessConfig(),
                  ValidationConfig(n_folds=3, inner_folds=3), n_iter=2)
    assert "roc_auc_ovr" in set(r.summary["metric"])
    d = prepare_xy(regression_df, design_for(regression_df, "response", "regression"))
    r = nested_cv(d, np.arange(len(d.y)), "regression", ["ridge", "lasso"], PreprocessConfig(),
                  ValidationConfig(cv_strategy="kfold", n_folds=3, inner_folds=3), n_iter=3)
    assert r.summary.set_index("metric").loc["r2", "mean"] > 0.5


def test_tune_improves_or_matches(bin_data):
    r = tune(bin_data, np.arange(240), "binary", "rf", PreprocessConfig(), ValidationConfig(), n_iter=5)
    assert r.best_params and len(r.cv_results) == 5 and 0.5 < r.best_score <= 1


def test_holdout_evaluation_and_ledger(bin_data):
    split = holdout_split(bin_data.y, "binary", 0.25, None, 0)
    pipe = fit_final(bin_data, split.train_idx, "binary", "lr", PreprocessConfig(), ValidationConfig())
    led = HoldoutLedger()
    r = evaluate_holdout(pipe, bin_data, split.test_idx, "binary", n_boot=300, ledger=led, model_desc="lr")
    m = r["metrics"].set_index("metric")
    assert m.loc["roc_auc", "ci_low"] < m.loc["roc_auc", "estimate"] < m.loc["roc_auc", "ci_high"]
    assert led.warning() is None
    evaluate_holdout(pipe, bin_data, split.test_idx, "binary", n_boot=50, ledger=led)
    assert "2 times" in led.warning()


def test_cluster_bootstrap_is_wider_than_naive():
    """Correlated rows within groups: resampling groups gives a wider (honest) CI."""
    rng = np.random.default_rng(0)
    g = np.repeat(np.arange(30), 10)
    gp = rng.normal(0, 1, 30)
    y = np.repeat(rng.binomial(1, 0.5, 30), 10)
    score = gp[g] + 0.8 * y + rng.normal(0, 0.1, 300)
    proba = np.c_[1 - 1 / (1 + np.exp(-score)), 1 / (1 + np.exp(-score))]
    naive = bootstrap_ci("binary", y, (proba[:, 1] > .5).astype(int), proba, 2, None, 500)
    clus = bootstrap_ci("binary", y, (proba[:, 1] > .5).astype(int), proba, 2, g, 500)
    w = lambda t: (t.set_index("metric").loc["roc_auc", "ci_high"] - t.set_index("metric").loc["roc_auc", "ci_low"])
    assert w(clus) > 1.8 * w(naive)
    assert clus.attrs["unit"] == "group" and clus.attrs["n_units"] == 30


def test_external_validation_compatibility_and_shift():
    dev, ext = synthetic.external_pair(shift=1.0)
    dsg = design_for(dev, "outcome", "binary", sample_id="sample_id")
    d = prepare_xy(dev, dsg)
    pipe = fit_final(d, np.arange(len(d.y)), "binary", "lr", PreprocessConfig(), ValidationConfig())
    r = external_validate(pipe, d, ext, dsg, "binary", n_boot=200)
    top = r.shift["features"].head(5)["feature"].tolist()
    assert set(top) == {"f0006", "f0007", "f0008", "f0009", "f0010"}
    assert r.shift["domain_auc"] > 0.8
    assert r.metrics.set_index("metric").loc["roc_auc", "estimate"] > 0.75
    with pytest.raises(ValueError, match="lack model features"):
        external_validate(pipe, d, ext.drop(columns=["f0001"]), dsg, "binary")
    r2 = external_validate(pipe, d, ext.drop(columns=["f0001"]), dsg, "binary",
                           allow_missing_features=True, n_boot=100)
    assert r2.compatibility.missing_features == ["f0001"]
    r3 = external_validate(pipe, d, ext.drop(columns=["outcome"]), dsg, "binary")
    assert r3.metrics is None and "P(case)" in r3.predictions
