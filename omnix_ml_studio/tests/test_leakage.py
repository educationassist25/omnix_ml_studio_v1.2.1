"""Mandatory leakage tests.

Each test either (a) traces exactly which rows every fit/predict call sees, using the
Spy classifier, or (b) runs a deliberately constructed leakage scenario to show that
the safeguard removes optimism that a naive analysis would report.
"""
import joblib
import numpy as np
import pytest
from conftest import design_for
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

import spy
from ml_studio.core import classification, synthetic
from ml_studio.core.classification import ModelSpec
from ml_studio.core.config import PreprocessConfig, ValidationConfig
from ml_studio.core.model_comparison import compare_models
from ml_studio.core.preprocessing import prepare_xy
from ml_studio.core.splitting import holdout_split
from ml_studio.core.validation import external_validate, fit_final, nested_cv

RAW = PreprocessConfig(scaling="none", max_missing_fraction=None, variance_threshold=None,
                       numeric_imputation="none")


@pytest.fixture
def spy_model(monkeypatch):
    spy.LOG.clear()
    monkeypatch.setitem(classification.REGISTRY, "spy",
                        ModelSpec("spy", "Spy", lambda rs: spy.Spy(), {"model__alpha": [0.1, 1.0, 10.0]}))
    yield
    spy.LOG.clear()


def fold_pairs(log):
    """[(fit_ids, [predict_ids, ...]), ...]: each fitted model instance with the rows it predicted.

    Paired by model instance, not call order, because folds may run in parallel threads."""
    out, by_model = [], {}
    for e in log:
        if e[0] == "fit":
            by_model[e[4]] = (e[1], [])
            out.append(by_model[e[4]])
        elif e[0] == "predict" and e[4] in by_model:
            by_model[e[4]][1].append(e[1])
    return out


def _data(df=None, target="outcome", **kw):
    df = spy.with_rid(df if df is not None else synthetic.binary(n=200, missing=0, with_categorical=False))
    return prepare_xy(df, design_for(df, target, "binary", **kw))


# 1 ------------------------------------------------------------------ holdout isolation
def test_holdout_rows_never_seen_during_development(spy_model):
    d = _data()
    split = holdout_split(d.y, "binary", 0.2, None, 0)
    hold = set(split.test_idx.tolist())
    compare_models(d, "binary", ["spy"], RAW, ValidationConfig(), train_idx=split.train_idx)
    fit_final(d, split.train_idx, "binary", "spy", RAW, ValidationConfig())
    seen = set().union(*[e[1] for e in spy.LOG])
    assert seen and not (seen & hold)


def test_holdout_values_do_not_affect_learned_preprocessing():
    df = synthetic.binary(n=200, missing=0.05)
    d = prepare_xy(df, design_for(df, "outcome", "binary"))
    split = holdout_split(d.y, "binary", 0.2, None, 0)
    cfg = PreprocessConfig(scaling="standard", numeric_imputation="median", correlation_threshold=0.9,
                           univariate_k=8)
    a = fit_final(d, split.train_idx, "binary", "lr", cfg, ValidationConfig())
    # corrupt the holdout rows drastically, refit on the same training rows
    d.X.iloc[split.test_idx, 2:] = 1e6
    b = fit_final(d, split.train_idx, "binary", "lr", cfg, ValidationConfig())
    sa = a.named_steps["columns"].named_transformers_["num"].named_steps["scale"]
    sb = b.named_steps["columns"].named_transformers_["num"].named_steps["scale"]
    assert np.allclose(sa.mean_, sb.mean_) and np.allclose(sa.scale_, sb.scale_)
    assert (a.named_steps["select"].get_support() == b.named_steps["select"].get_support()).all()
    assert np.allclose(a.named_steps["model"].coef_, b.named_steps["model"].coef_)


# 2 ------------------------------------------------------------------ feature selection
def test_feature_selection_inside_folds_removes_optimism():
    """Pure-noise omics (p=2000, n=80). Selecting top features on all data before CV
    (naive) gives a strongly inflated AUC; the engine's in-fold selection gives ~0.5."""
    df = synthetic.omics_hdlss(n=80, p=2000, null=True, seed=11)
    d = prepare_xy(df, design_for(df, "group", "binary"))
    X = np.log2(d.X.values + 1)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    naive_X = SelectKBest(f_classif, k=20).fit_transform(X, d.y)          # leakage!
    naive = cross_val_score(LogisticRegression(max_iter=2000), naive_X, d.y, cv=cv,
                            scoring="roc_auc").mean()
    cfg = PreprocessConfig(transform="log2", log_offset=1.0, univariate_k=20)
    safe = compare_models(d, "binary", ["lr"], cfg,
                          ValidationConfig(cv_strategy="repeated_stratified_kfold", n_repeats=3))
    safe_auc = safe.leaderboard["roc_auc_mean"].iloc[0]
    assert naive > 0.85, naive
    assert 0.3 < safe_auc < 0.65, safe_auc


def test_selector_fitted_only_on_training_rows(spy_model):
    d = _data()
    cfg = PreprocessConfig(scaling="none", max_missing_fraction=None, variance_threshold=None,
                           numeric_imputation="none", univariate_k=30)   # keeps rid (k >= p)
    r = compare_models(d, "binary", ["spy"], cfg, ValidationConfig(n_folds=5))
    pairs = fold_pairs(spy.LOG)
    assert len(pairs) == 5
    for fit_ids, pred_sets in pairs:
        assert pred_sets and all(not (fit_ids & p) for p in pred_sets)
    assert r.failures == {}


# 3 ------------------------------------------------------------------ resampling
@pytest.mark.parametrize("mode", ["oversample", "smote", "undersample"])
def test_resampling_only_in_training_folds(spy_model, mode):
    df = spy.with_rid(synthetic.imbalanced(n=300, minority=0.1))
    d = prepare_xy(df, design_for(df, "event", "binary"))
    cfg = PreprocessConfig(scaling="none", max_missing_fraction=None, variance_threshold=None,
                           numeric_imputation="none", imbalance=mode)
    compare_models(d, "binary", ["spy"], cfg, ValidationConfig(n_folds=5))
    fits = [e for e in spy.LOG if e[0] == "fit"]
    preds = [e for e in spy.LOG if e[0] == "predict"]
    assert all(f[3][0] == f[3][1] for f in fits)              # training folds balanced
    test_sizes = {len(p[1]) for p in preds}
    assert test_sizes <= {60}                                 # test folds untouched (300 / 5)
    all_pred = set().union(*[p[1] for p in preds])
    assert all_pred == set(range(300))                        # every real row predicted, no synthetic


# 4 ------------------------------------------------------------------ groups
def test_grouped_cv_removes_patient_fingerprint_leakage():
    """Repeated visits, outcome random per patient: true AUC = 0.5. Row-wise CV lets
    the model recognise patients across folds; grouped CV does not."""
    df = synthetic.grouped_repeated()
    dsg = design_for(df, "response", "binary", exclude=("sample_id", "patient_id", "visit"))
    leaky = compare_models(prepare_xy(df, dsg), "binary", ["knn"], PreprocessConfig(),
                           ValidationConfig(cv_strategy="stratified_kfold"))
    dsg.group = "patient_id"
    safe = compare_models(prepare_xy(df, dsg), "binary", ["knn"], PreprocessConfig(),
                          ValidationConfig(cv_strategy="stratified_group_kfold", n_repeats=3))
    assert leaky.leaderboard["roc_auc_mean"].iloc[0] > 0.9
    assert 0.35 < safe.leaderboard["roc_auc_mean"].iloc[0] < 0.65


def test_group_rows_not_split_in_any_fit(spy_model):
    df = spy.with_rid(synthetic.grouped_repeated(n_patients=40))
    d = prepare_xy(df, design_for(df, "response", "binary",
                                  exclude=("sample_id", "patient_id", "visit"), group="patient_id"))
    gid = dict(zip(range(len(df)), df["patient_id"]))
    compare_models(d, "binary", ["spy"], RAW, ValidationConfig(cv_strategy="stratified_group_kfold"))
    pairs = fold_pairs(spy.LOG)
    assert len(pairs) == 5
    for fit_ids, pred_sets in pairs:
        fg = {gid[i] for i in fit_ids}
        for p in pred_sets:
            assert not (fg & {gid[i] for i in p})


# 5 ------------------------------------------------------------------ nested CV
def test_tuning_never_sees_outer_test_fold(spy_model):
    d = _data()
    markers = []

    def mark(k, n, msg):
        spy.LOG.append(("outer", k, None, None)); markers.append(k)

    res = nested_cv(d, np.arange(len(d.y)), "binary", ["spy"], RAW,
                    ValidationConfig(n_folds=4, inner_folds=3), n_iter=3, progress=mark)
    assert markers == [0, 1, 2, 3]
    cur = None
    for e in spy.LOG:
        if e[0] == "outer":
            cur = e[1]; outer_test = set(res.outer_splits[cur][1].tolist())
            continue
        if e[0] == "fit":
            assert not (e[1] & outer_test), f"outer fold {cur}: tuning saw outer-test rows"


# 6 ------------------------------------------------------------------ external
def test_external_validation_does_not_refit(monkeypatch):
    dev, ext = synthetic.external_pair()
    dsg = design_for(dev, "outcome", "binary", sample_id="sample_id")
    d = prepare_xy(dev, dsg)
    pipe = fit_final(d, np.arange(len(d.y)), "binary", "lr", PreprocessConfig(), ValidationConfig())
    before = joblib.hash(pipe)

    def boom(*a, **k):
        raise AssertionError("fit called during external validation")
    # patch only this fitted pipeline's own objects (the separate dataset-shift
    # classifier is a different model and may be fitted)
    for obj in [pipe] + [st for _, st in pipe.steps]:
        for meth in ("fit", "fit_transform"):
            if hasattr(obj, meth):
                monkeypatch.setattr(obj, meth, boom)
    r = external_validate(pipe, d, ext, dsg, "binary", n_boot=200)     # (a) no fit call
    monkeypatch.undo()
    # (b) a fresh, unpatched fit (deterministic: same hash) is unchanged afterwards
    clean = fit_final(d, np.arange(len(d.y)), "binary", "lr", PreprocessConfig(), ValidationConfig())
    h = joblib.hash(clean)
    assert h == before
    external_validate(clean, d, ext, dsg, "binary", n_boot=50)
    assert joblib.hash(clean) == h
    assert r.metrics is not None and len(r.predictions) == len(ext)
