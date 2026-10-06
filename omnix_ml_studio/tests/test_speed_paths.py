"""The fast paths (fold cache, isolated workers, cached tuning) must give exactly the same scores."""
import numpy as np
import pandas as pd
import pytest
from conftest import design_for

from ml_studio.core import synthetic as SY
from ml_studio.core.config import PreprocessConfig, ValidationConfig
from ml_studio.core.model_comparison import compare_models, create_model
from ml_studio.core.optimization import tune
from ml_studio.core.preprocessing import prepare_xy

CASES = [("binary", SY.binary, "outcome", ["dummy", "lr", "rf", "knn", "lightgbm", "nb", "svm"]),
         ("multiclass", SY.multiclass, "subtype", ["dummy", "lr", "rf", "ridge", "dt"]),
         ("regression", SY.regression, "response", ["dummy", "lr", "ridge", "rf", "knn", "svm"])]


def _scores(lb, keys):
    t = lb.set_index("key").loc[keys]
    return t[[c for c in t.columns if c.endswith("_mean")]].to_numpy(float)


@pytest.mark.parametrize("task,make,target,keys", CASES)
@pytest.mark.parametrize("prep", [PreprocessConfig(), PreprocessConfig(univariate_k=8, imbalance="smote"),
                                  PreprocessConfig(imbalance="class_weight", fs_method="l1", fs_k=6)])
def test_cache_and_isolation_match_pipelines(task, make, target, keys, prep):
    if task == "regression" and prep.imbalance != "none":
        prep = PreprocessConfig(univariate_k=8)
    df = make()
    d = prepare_xy(df, design_for(df, target, task))
    v = ValidationConfig(cv_strategy="kfold" if task == "regression" else "stratified_kfold")
    ref = _scores(compare_models(d, task, keys, prep, v, use_cache=False).leaderboard, keys)
    for kw in ({"use_cache": True}, {"isolate": True, "workers": 2}):
        got = _scores(compare_models(d, task, keys, prep, v, **kw).leaderboard, keys)
        assert (np.isnan(ref) == np.isnan(got)).all(), kw          # NaN only where the metric is undefined
        assert np.isfinite(ref).mean() > 0.7
        np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-12, equal_nan=True)


def test_cached_tuning_and_create_match():
    df = SY.binary()
    d = prepare_xy(df, design_for(df, "outcome", "binary"))
    idx = np.arange(len(d.y))
    prep, v = PreprocessConfig(univariate_k=10), ValidationConfig()
    tuned = {}
    for key in ("rf", "lr", "knn"):
        a = tune(d, idx, "binary", key, prep, v, n_iter=6, use_cache=False)
        b = tune(d, idx, "binary", key, prep, v, n_iter=6, use_cache=True)
        assert a.best_params == b.best_params and a.best_score == pytest.approx(b.best_score, abs=1e-12)
        assert len(b.cv_results) == 6 and np.isfinite(b.cv_results["mean_test_score"]).all()
        tuned[key] = b
    cm = create_model(d, idx, "binary", "rf", prep, v, tuned["rf"].best_params)
    lb = compare_models(d, "binary", ["rf"], prep, v, train_idx=idx).leaderboard
    assert cm.cv_signature and np.isfinite(cm.folds.to_numpy(float)).all()
    assert lb["roc_auc_mean"].iloc[0] > 0.5


def test_time_limit_skips_early_and_stop_works():
    df = SY.binary()
    d = prepare_xy(df, design_for(df, "outcome", "binary"))
    v = ValidationConfig()
    r = compare_models(d, "binary", ["dummy", "rf"], PreprocessConfig(), v, per_model_timeout_s=1e-4)
    assert "rf" in r.failures and "projected" in r.failures["rf"]
    r = compare_models(d, "binary", ["dummy", "lr", "rf"], PreprocessConfig(), v, should_stop=lambda: True)
    assert r.leaderboard.empty and all("stopped" in m for m in r.failures.values())
    seen = []
    r = compare_models(d, "binary", ["dummy", "lr"], PreprocessConfig(), v, isolate=True, workers=1,
                       on_result=lambda row: seen.append(row["key"]))
    assert sorted(seen) == ["dummy", "lr"] and len(r.leaderboard) == 2


def test_gaussian_process_learns_on_wide_data():
    """Library-default GP predicts the prior on 3,000 standardised features (AUC 0.5); the scaled one learns."""
    rng = np.random.default_rng(0)
    n, p = 120, 3000
    y = rng.integers(0, 2, n)
    X = rng.normal(size=(n, p)) + np.outer(y, np.r_[np.full(300, 1.0), np.zeros(p - 300)])
    df = pd.DataFrame(X, columns=[f"g{i}" for i in range(p)]).assign(outcome=np.where(y, "b", "a"))
    d = prepare_xy(df, design_for(df, "outcome", "binary", exclude=()))
    lb = compare_models(d, "binary", ["gpc"], PreprocessConfig(), ValidationConfig(n_folds=3)).leaderboard
    assert lb["roc_auc_mean"].iloc[0] > 0.9


def test_wide_catboost_settings():
    pytest.importorskip("catboost")
    from sklearn.base import clone

    from ml_studio.core import classification as C
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 900))
    y = (X[:, 0] > 0).astype(int)
    m = clone(C.REGISTRY["catboost"].factory(0)).set_params(iterations=20)
    m.fit(X, y)
    assert m.get_params()["border_count"] == 32 and 0 < m.get_params()["rsm"] < 1
    small = clone(C.REGISTRY["catboost"].factory(0)).set_params(iterations=20).fit(X[:, :50], y)
    assert small.get_params().get("border_count") is None          # defaults kept on narrow data


@pytest.mark.parametrize("prep", [PreprocessConfig(univariate_k=10), PreprocessConfig(transform="yeo-johnson"),
                                  PreprocessConfig(numeric_imputation="knn")])
def test_fast_permutation_importance_is_identical(prep):
    from ml_studio.core.explainability import permutation_importances
    from ml_studio.core.validation import fit_final
    df = SY.binary()
    d = prepare_xy(df, design_for(df, "outcome", "binary"))
    pipe = fit_final(d, np.arange(len(d.y)), "binary", "rf", prep, ValidationConfig())
    slow = permutation_importances(pipe, d.X, d.y, "binary", 2, 4, 0).set_index("feature")["importance_mean"]
    fast = permutation_importances(pipe, d.X, d.y, "binary", 2, 4, 0,
                                   categorical=d.categorical).set_index("feature")["importance_mean"]
    pd.testing.assert_series_equal(slow.sort_index(), fast.sort_index())
