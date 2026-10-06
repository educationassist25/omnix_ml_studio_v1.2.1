import time

import numpy as np
import pandas as pd
import pytest
from conftest import design_for
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

from ml_studio.core import synthetic
from ml_studio.core.config import PreprocessConfig, ValidationConfig
from ml_studio.core.feature_selection import (METHODS, FeatureSelector, kuncheva, selected_features,
                                              selection_stability, to_original, univariate_statistics)
from ml_studio.core.model_comparison import compare_models
from ml_studio.core.preprocessing import build_preprocessor, prepare_xy
from ml_studio.core.validation import fit_final

INFORMATIVE = {"f0001", "f0002", "f0003", "f0004", "f0005"}
SUPERVISED = [m for m in METHODS if m != "variance"]


@pytest.fixture(scope="module")
def signal():
    df = synthetic.binary(n=300, p=40, n_informative=5, seed=4, missing=0, with_categorical=False, effect=1.2)
    d = prepare_xy(df, design_for(df, "outcome", "binary"))
    Z = build_preprocessor(d.numeric, [], PreprocessConfig(), "binary").fit_transform(d.X)
    return d, Z


@pytest.mark.parametrize("method", SUPERVISED)
def test_supervised_methods_recover_informative_features(signal, method):
    d, Z = signal
    fs = FeatureSelector(method, k=8, task="binary", random_state=0).fit(Z, d.y)
    chosen = set(np.asarray(Z.columns)[fs.get_support()])
    hits = len(chosen & INFORMATIVE)
    # the k-NN mutual-information estimator is high-variance at this n / effect size
    assert hits >= (3 if method == "mutual_info" else 4), (method, sorted(chosen))
    if method != "rfecv":
        assert fs.get_support().sum() == 8
    out = fs.set_output(transform="pandas").transform(Z)
    assert list(out.columns) == list(np.asarray(Z.columns)[fs.get_support()])


@pytest.mark.parametrize("method", ["univariate", "mutual_info", "l1", "model", "tree", "rfe"])
def test_regression_selection(regression_df, method):
    d = prepare_xy(regression_df, design_for(regression_df, "response", "regression"))
    Z = build_preprocessor(d.numeric, [], PreprocessConfig(), "regression").fit_transform(d.X)
    fs = FeatureSelector(method, k=6, task="regression").fit(Z, d.y)
    assert len(set(np.asarray(Z.columns)[fs.get_support()]) & INFORMATIVE) >= 4


def test_multiclass_selection(multiclass_df):
    d = prepare_xy(multiclass_df, design_for(multiclass_df, "subtype", "multiclass"))
    Z = build_preprocessor(d.numeric, [], PreprocessConfig(), "multiclass").fit_transform(d.X)
    for m in ("univariate", "l1", "tree", "rfe"):
        fs = FeatureSelector(m, k=6, task="multiclass").fit(Z, d.y)
        assert len(set(np.asarray(Z.columns)[fs.get_support()]) & {f"f000{i}" for i in range(1, 7)}) >= 4, m


def test_k_larger_than_p_and_permutation_limit(signal):
    d, Z = signal
    assert FeatureSelector("univariate", k=999).fit(Z, d.y).get_support().all()
    big = pd.DataFrame(np.random.default_rng(0).normal(size=(60, 400)))
    big.columns = [f"x{i}" for i in range(400)]
    with pytest.raises(ValueError, match="pre-filter"):
        FeatureSelector("permutation", 10).fit(big, np.r_[np.zeros(30), np.ones(30)].astype(int))


@pytest.mark.parametrize("method", ["l1", "tree", "rfe", "mutual_info"])
def test_in_pipeline_selection_is_not_optimistic_on_noise(method):
    """Pure-noise HDLSS: in-fold selection must give chance-level CV AUC."""
    df = synthetic.omics_hdlss(n=80, p=1000, null=True, seed=31)
    d = prepare_xy(df, design_for(df, "group", "binary"))
    cfg = PreprocessConfig(transform="log2", log_offset=1.0, univariate_k=200, fs_method=method, fs_k=15)
    r = compare_models(d, "binary", ["lr"], cfg, ValidationConfig(cv_strategy="repeated_stratified_kfold",
                                                                    n_repeats=2))
    assert r.failures == {}
    assert 0.3 < r.leaderboard["roc_auc_mean"].iloc[0] < 0.68


def test_pipeline_selection_used_and_recoverable(signal):
    d, _ = signal
    cfg = PreprocessConfig(fs_method="l1", fs_k=6)
    pipe = fit_final(d, np.arange(len(d.y)), "binary", "lr", cfg, ValidationConfig())
    feats = selected_features(pipe)
    assert len(feats) == 6 and len(set(feats) & INFORMATIVE) >= 4
    assert pipe.named_steps["model"].coef_.shape[1] == 6


def test_stability_signal_vs_null():
    df = synthetic.binary(n=200, p=60, n_informative=5, seed=8, missing=0, with_categorical=False, effect=1.2)
    d = prepare_xy(df, design_for(df, "outcome", "binary"))
    r = selection_stability(d, np.arange(200), "binary", ["univariate", "l1", "tree"], 5, PreprocessConfig(),
                            ValidationConfig(n_folds=5), n_repeats=3)
    assert r.settings["n_resamples"] == 15
    top = set(r.frequency.index[:5])
    assert len(top & INFORMATIVE) >= 4
    assert (r.frequency.loc[list(INFORMATIVE), "univariate"] >= 0.8).sum() >= 4
    k_sig = r.indices.set_index("method").loc["univariate", "kuncheva"]
    null = synthetic.binary(n=200, p=60, n_informative=5, seed=8, missing=0, with_categorical=False, effect=0.0)
    dn = prepare_xy(null, design_for(null, "outcome", "binary"))
    rn = selection_stability(dn, np.arange(200), "binary", ["univariate"], 5, PreprocessConfig(),
                             ValidationConfig(n_folds=5), n_repeats=3)
    k_null = rn.indices.set_index("method").loc["univariate", "kuncheva"]
    assert k_sig > 0.7 and k_null < 0.35, (k_sig, k_null)
    assert r.overlap.shape == (3, 3) and np.allclose(np.diag(r.overlap), 1)


def test_stability_grouped_respects_groups(grouped_df):
    d = prepare_xy(grouped_df, design_for(grouped_df, "response", "binary",
                                          exclude=("sample_id", "patient_id", "visit"), group="patient_id"))
    r = selection_stability(d, np.arange(len(d.y)), "binary", ["univariate"], 5, PreprocessConfig(),
                            ValidationConfig(cv_strategy="stratified_group_kfold", n_folds=4), n_repeats=2)
    assert r.settings["n_resamples"] == 8


def test_kuncheva_bounds():
    assert kuncheva([{1, 2, 3}, {1, 2, 3}], 3, 100) == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    sets = [set(rng.choice(1000, 10, replace=False)) for _ in range(30)]
    assert abs(kuncheva(sets, 10, 1000)) < 0.05


def test_univariate_statistics_match_scipy_and_statsmodels(binary_df):
    d = prepare_xy(binary_df, design_for(binary_df, "outcome", "binary"))
    st_ = univariate_statistics(d, np.arange(len(d.y)), "binary").set_index("feature")
    x = d.X["f0003"].to_numpy(float); ok = np.isfinite(x)
    p = mannwhitneyu(x[ok][d.y[ok] == 1], x[ok][d.y[ok] == 0]).pvalue
    assert st_.loc["f0003", "p"] == pytest.approx(p)
    assert np.allclose(st_["q_bh"].sort_index(), multipletests(st_["p"].sort_index(), method="fdr_bh")[1])
    assert set(st_.index) == set(d.numeric) | set(d.categorical)
    assert st_.loc["stage", "test"].startswith("χ²")


def test_univariate_statistics_regression_multiclass(regression_df, multiclass_df):
    d = prepare_xy(regression_df, design_for(regression_df, "response", "regression"))
    assert univariate_statistics(d, np.arange(len(d.y)), "regression")["feature"].iloc[0] in INFORMATIVE
    d = prepare_xy(multiclass_df, design_for(multiclass_df, "subtype", "multiclass"))
    assert univariate_statistics(d, np.arange(len(d.y)), "multiclass")["test"].iloc[0].startswith("Kruskal")


def test_to_original():
    assert to_original("stage_III", ["stage", "sex"]) == "stage"
    assert to_original("sex_M", ["stage", "sex"]) == "sex"
    assert to_original("f0001", ["stage"]) == "f0001"
    assert to_original("site_a_b", ["site", "site_a"]) == "site_a"


def test_hdlss_selection_performance():
    """2,000 features × 80 samples: stability over 10 resamples within a time budget."""
    df = synthetic.omics_hdlss(n=80, p=2000)
    d = prepare_xy(df, design_for(df, "group", "binary"))
    t = time.time()
    r = selection_stability(d, np.arange(80), "binary", ["univariate", "l1", "tree"], 20,
                            PreprocessConfig(transform="log2", log_offset=1.0), ValidationConfig(n_folds=5),
                            n_repeats=2)
    dt = time.time() - t
    assert dt < 120, dt
    informative = {f"m{i:05d}" for i in range(1, 11)}
    assert len(set(r.frequency.index[:10]) & informative) >= 8
