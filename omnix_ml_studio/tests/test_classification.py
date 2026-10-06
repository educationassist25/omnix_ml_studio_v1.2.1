import numpy as np
import pytest
from conftest import design_for

from ml_studio.core import classification
from ml_studio.core.config import PreprocessConfig, ValidationConfig
from ml_studio.core.model_comparison import compare_models, pretty_leaderboard
from ml_studio.core.preprocessing import prepare_xy


@pytest.fixture(scope="module")
def bin_result(binary_df):
    d = prepare_xy(binary_df, design_for(binary_df, "outcome", "binary"))
    return compare_models(d, "binary", list(classification.REGISTRY), PreprocessConfig(),
                          ValidationConfig(n_folds=5))


def test_every_registered_classifier_runs(bin_result):
    assert bin_result.failures == {}
    assert set(bin_result.leaderboard["key"]) == set(classification.REGISTRY)


def test_models_beat_dummy_and_dummy_is_chance(bin_result):
    lb = bin_result.leaderboard.set_index("key")
    assert abs(lb.loc["dummy", "roc_auc_mean"] - 0.5) < 1e-9
    assert lb.loc["lr", "roc_auc_mean"] > 0.75
    assert lb.loc["lr", "beats_baseline"]


def test_leaderboard_has_all_binary_metrics(bin_result):
    for m in ("roc_auc", "average_precision", "accuracy", "balanced_accuracy", "precision",
              "recall", "f1", "mcc", "log_loss"):
        assert f"{m}_mean" in bin_result.leaderboard
    assert (bin_result.leaderboard["log_loss_mean"].dropna() > 0).all()   # sign flipped back; NaN = no probabilities
    assert "Beats dummy" in pretty_leaderboard(bin_result.leaderboard, "binary")


def test_multiclass(multiclass_df):
    d = prepare_xy(multiclass_df, design_for(multiclass_df, "subtype", "multiclass"))
    r = compare_models(d, "multiclass", ["dummy", "lr", "rf", "lightgbm"], PreprocessConfig(),
                       ValidationConfig())
    lb = r.leaderboard.set_index("key")
    assert r.failures == {}
    assert abs(lb.loc["dummy", "roc_auc_ovr_mean"] - 0.5) < 1e-9
    assert lb.loc["lr", "roc_auc_ovr_mean"] > 0.8
    assert "f1_macro_mean" in lb


@pytest.mark.parametrize("mode", ["class_weight", "oversample", "undersample", "smote"])
def test_imbalance_modes(mode):
    from ml_studio.core import synthetic
    df = synthetic.imbalanced(n=300, minority=0.1)
    d = prepare_xy(df, design_for(df, "event", "binary"))
    base = compare_models(d, "binary", ["lr"], PreprocessConfig(), ValidationConfig())
    r = compare_models(d, "binary", ["lr", "rf"], PreprocessConfig(imbalance=mode), ValidationConfig())
    assert r.failures == {}
    lr = r.leaderboard.set_index("key").loc["lr"]
    # minority recall must improve over no handling (default 0.5 threshold)
    assert lr["recall_mean"] > base.leaderboard["recall_mean"].iloc[0] + 0.2


def test_reproducible_with_seed(binary_df):
    d = prepare_xy(binary_df, design_for(binary_df, "outcome", "binary"))
    a = compare_models(d, "binary", ["rf", "lightgbm"], PreprocessConfig(), ValidationConfig(random_state=7))
    b = compare_models(d, "binary", ["rf", "lightgbm"], PreprocessConfig(), ValidationConfig(random_state=7))
    assert a.cv_signature == b.cv_signature
    assert np.allclose(a.leaderboard["roc_auc_mean"], b.leaderboard["roc_auc_mean"])
    c = compare_models(d, "binary", ["rf"], PreprocessConfig(), ValidationConfig(random_state=8))
    assert c.cv_signature != a.cv_signature


def test_failing_model_reported_not_crashing(omics_df):
    d = prepare_xy(omics_df, design_for(omics_df, "group", "binary"))
    r = compare_models(d, "binary", ["lr"], PreprocessConfig(transform="log2"), ValidationConfig())
    assert "lr" in r.failures and "explicit" in r.failures["lr"]


def test_full_pycaret_list_present():
    from ml_studio.core import regression
    clf = {"lr", "knn", "nb", "dt", "svm", "rbfsvm", "gpc", "mlp", "ridge", "rf", "qda", "ada", "gbc", "lda",
           "et", "xgboost", "lightgbm", "catboost", "dummy"}
    reg = {"lr", "lasso", "ridge", "en", "lar", "llar", "omp", "br", "ard", "par", "ransac", "tr", "huber", "kr",
           "svm", "knn", "dt", "rf", "et", "ada", "gbr", "mlp", "xgboost", "lightgbm", "catboost", "dummy"}
    assert clf <= set(classification.REGISTRY), clf - set(classification.REGISTRY)
    assert reg <= set(regression.REGISTRY), reg - set(regression.REGISTRY)


def test_every_classifier_multiclass(multiclass_df):
    d = prepare_xy(multiclass_df, design_for(multiclass_df, "subtype", "multiclass"))
    r = compare_models(d, "multiclass", list(classification.REGISTRY), PreprocessConfig(),
                       ValidationConfig(n_folds=3))
    assert r.failures == {}, r.failures
    lb = r.leaderboard.set_index("key")
    assert np.isnan(lb.loc["ridge", "roc_auc_ovr_mean"]) and np.isnan(lb.loc["svm", "log_loss_mean"])
    assert lb.loc["ridge", "balanced_accuracy_mean"] > 0.6
    assert lb["roc_auc_ovr_mean"].notna().iloc[0]            # NaN-AUC models are ranked last


def test_no_proba_models_holdout_and_nested(binary_df):
    from ml_studio.core.splitting import holdout_split
    from ml_studio.core.validation import evaluate_holdout, fit_final, nested_cv
    d = prepare_xy(binary_df, design_for(binary_df, "outcome", "binary"))
    sp = holdout_split(d.y, "binary", 0.25, None, 0)
    for key in ("ridge", "svm"):
        pipe = fit_final(d, sp.train_idx, "binary", key, PreprocessConfig(), ValidationConfig())
        m = evaluate_holdout(pipe, d, sp.test_idx, "binary", n_boot=100)["metrics"].set_index("metric")
        assert m.loc["roc_auc", "estimate"] > 0.75 and "log_loss" not in m.index
    r = nested_cv(d, sp.train_idx, "binary", ["ridge", "lr"], PreprocessConfig(), ValidationConfig(n_folds=3),
                  n_iter=2)
    assert r.settings["selection_metric"] == "roc_auc"
