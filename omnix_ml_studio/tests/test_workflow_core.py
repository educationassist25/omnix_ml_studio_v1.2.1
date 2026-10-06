"""Core functions behind the Create, Analyze and Deploy steps."""
import numpy as np
import pandas as pd
import pytest
from conftest import design_for

from ml_studio.core import anomaly as AN
from ml_studio.core import clustering as CL
from ml_studio.core import model_manager as MM
from ml_studio.core import synthetic as SY
from ml_studio.core import timeseries as TS
from ml_studio.core.config import PreprocessConfig, ValidationConfig
from ml_studio.core.explainability import model_input
from ml_studio.core.model_comparison import compare_models, create_model
from ml_studio.core.optimization import tune
from ml_studio.core.preprocessing import prepare_xy
from ml_studio.core.splitting import holdout_split
from ml_studio.core.validation import cv_predictions, fit_final, learning_curve_scores


@pytest.fixture(scope="module")
def binary(binary_df):
    d = prepare_xy(binary_df, design_for(binary_df, "outcome", "binary"))
    sp = holdout_split(d.y, "binary", 0.2, None, 0)
    return d, sp


def test_create_model_uses_the_comparison_folds(binary):
    d, sp = binary
    v = ValidationConfig()
    cmp = compare_models(d, "binary", ["lr"], PreprocessConfig(), v, train_idx=sp.train_idx)
    cm = create_model(d, sp.train_idx, "binary", "lr", PreprocessConfig(), v)
    assert cm.cv_signature == cmp.cv_signature
    assert cm.summary["roc_auc"][0] == pytest.approx(cmp.leaderboard.set_index("key").loc["lr", "roc_auc_mean"])
    assert list(cm.folds.index[-2:]) == ["Mean", "SD"] and len(cm.folds) == v.n_folds + 2


def test_create_model_with_tuned_and_custom_params(binary):
    d, sp = binary
    v = ValidationConfig()
    t = tune(d, sp.train_idx, "binary", "rf", PreprocessConfig(), v, n_iter=3)
    cm = create_model(d, sp.train_idx, "binary", "rf", PreprocessConfig(), v, t.best_params, "tuned")
    assert cm.params == t.best_params
    custom = create_model(d, sp.train_idx, "binary", "lr", PreprocessConfig(), v, {"model__C": 0.001})
    default = create_model(d, sp.train_idx, "binary", "lr", PreprocessConfig(), v)
    assert custom.summary["log_loss"][0] != pytest.approx(default.summary["log_loss"][0])


def test_oof_proba_is_none_when_model_has_no_scores(multiclass_df):
    d = prepare_xy(multiclass_df, design_for(multiclass_df, "subtype", "multiclass"))
    idx = np.arange(len(d.y))
    _, yp, pr = cv_predictions(d, idx, "multiclass", "ridge", PreprocessConfig(), ValidationConfig())
    assert pr is None and len(yp) == len(idx)
    _, _, pr2 = cv_predictions(d, idx, "multiclass", "lr", PreprocessConfig(), ValidationConfig())
    assert pr2 is not None and np.allclose(pr2.sum(1), 1)


def test_learning_curve_shapes(binary):
    d, sp = binary
    sizes, tr, va, m = learning_curve_scores(d, sp.train_idx, "binary", "lr", PreprocessConfig(), ValidationConfig(),
                                             n_points=4)
    assert len(sizes) == 4 and tr.shape == va.shape and m == "roc_auc"
    assert np.all(np.diff(sizes) > 0) and sizes.max() < len(sp.train_idx)


def test_predict_new_flags_out_of_distribution(binary, binary_df):
    d, sp = binary
    pipe = fit_final(d, sp.train_idx, "binary", "lr", PreprocessConfig(), ValidationConfig())
    X = d.X.iloc[sp.train_idx]
    meta = {"task": "binary", "classes": d.classes, "categorical": d.categorical,
            "profile": MM.training_profile(X, d.numeric, d.categorical)}
    ood = MM.fit_ood_detector(model_input(pipe, X))
    new = binary_df.loc[d.index[sp.test_idx]].drop(columns=["outcome"]).head(10).copy()
    new.loc[new.index[:2], d.numeric] = 1e3
    out = MM.predict_new(pipe, meta, new, ood, "sample_id")
    assert list(out.columns[:2]) == ["sample_id", "predicted_class"]
    assert np.allclose(out[[f"P({c})" for c in d.classes]].sum(1), 1)
    assert (out["flag"].iloc[:2] != "").all() and (out["ood_percentile"].iloc[:2] > 99).all()
    direct = pipe.predict_proba(d.X.iloc[sp.test_idx[:10]])[:, 1]
    assert np.allclose(out[f"P({d.classes[1]})"].iloc[2:], direct[2:])


def test_predict_new_tolerates_missing_column(binary, binary_df):
    d, sp = binary
    pipe = fit_final(d, sp.train_idx, "binary", "lr", PreprocessConfig(), ValidationConfig())
    meta = {"task": "binary", "classes": d.classes, "categorical": d.categorical,
            "profile": MM.training_profile(d.X, d.numeric, d.categorical)}
    new = binary_df.drop(columns=["outcome", d.numeric[0]]).head(5)
    out = MM.predict_new(pipe, meta, new)
    assert out.attrs["check"]["missing"] == [d.numeric[0]] and len(out) == 5


def test_unsupervised_bundles_roundtrip():
    df = SY.clusters()
    feats = [c for c in df.columns if c.startswith("f")]
    ud = CL.prepare_unsupervised(df, feats)
    pre = CL.build_unsupervised_preprocessor(ud.numeric, ud.categorical, PreprocessConfig())
    Z = pd.DataFrame(np.asarray(pre.fit_transform(ud.X), float))
    model, lab = CL.fit_predict(Z, "kmeans", {"n_clusters": 3})
    meta = {"module": "clustering", "categorical": [], "profile": MM.training_profile(ud.X, ud.numeric, [])}
    obj, _, meta2 = MM.load_bundle(MM.make_bundle({"pre": pre, "model": model, "key": "kmeans", "Z": Z, "labels": lab}, meta))
    Zn = pd.DataFrame(np.asarray(obj["pre"].transform(MM.prepare_features(df.head(20), meta2)), float))
    lab2, how = CL.assign_new(obj["model"], "kmeans", obj["Z"], obj["labels"], Zn)
    assert how == "native predict" and np.array_equal(lab2, lab[:20])

    a = SY.anomalies()
    af = [c for c in a.columns if c.startswith("f")]
    uda = CL.prepare_unsupervised(a, af)
    prea = CL.build_unsupervised_preprocessor(uda.numeric, [], PreprocessConfig())
    Za = pd.DataFrame(np.asarray(prea.fit_transform(uda.X), float))
    det, sc, _ = AN.fit_detector(Za, "iforest", 0.05)
    obj, _, _ = MM.load_bundle(MM.make_bundle({"pre": prea, "model": det, "key": "iforest"}, {"module": "anomaly"}))
    s2, _ = AN.score_new(obj["model"], pd.DataFrame(np.asarray(obj["pre"].transform(uda.X), float)), "iforest")
    assert np.corrcoef(s2, sc)[0, 1] > 0.99


def test_forecaster_bundle_forecasts_identically():
    y = TS.prepare_series(SY.monthly_series(), "value", "date").y
    design = TS.TSDesign(12, 12, 2, True)
    r = TS.final_forecast(y, "theta", design, horizon=6)
    mdl, _, meta = MM.load_bundle(MM.make_bundle(r["model"], {"module": "timeseries"}))
    p = mdl.predict(fh=list(range(1, 7)))
    assert np.allclose(p.values, r["table"]["forecast"].values) and meta["module"] == "timeseries"


def test_reference_label_is_never_a_feature():
    df = SY.clusters()
    feats = [c for c in df.columns if c.startswith("f")]
    ud = CL.prepare_unsupervised(df, feats, reference="true_subtype", sample_id="sample_id")
    assert "true_subtype" not in ud.X.columns and "sample_id" not in ud.X.columns
    assert np.array_equal(ud.reference, df["true_subtype"].to_numpy())


def test_transductive_detectors_cannot_score_new_data():
    a = SY.anomalies(n=200)
    Z = a[[c for c in a.columns if c.startswith("f")]]
    for k in ("cof", "sod", "sos"):
        m, _, _ = AN.fit_detector(Z, k, 0.05)
        with pytest.raises(AN.TransductiveError):
            AN.score_new(m, Z.head(5), k)


def test_cv_never_sees_the_future():
    """A forecaster that memorises its training data must not know the validation values."""
    y = TS.prepare_series(SY.monthly_series(), "value", "date").y
    design = TS.TSDesign(12, 12, 3, True)
    n_dev, folds = design.split(len(y))
    seen = []
    orig = TS._fit_predict

    def spy(key, y_train, *a, **k):
        seen.append(y_train.index[-1])
        return orig(key, y_train, *a, **k)
    TS._fit_predict = spy
    try:
        TS.cross_validate_forecaster(y, "naive", design)
    finally:
        TS._fit_predict = orig
    assert [str(s) for s in seen] == [str(y.index[a - 1]) for a, _ in folds]
    assert all(b <= n_dev for _, b in folds)


def test_comparison_views():
    from ml_studio.core import comparisons as CMP
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 200)
    P = rng.dirichlet([1, 1, 1], 200)
    yb, P2, lab, mask = CMP.one_vs_rest(y, P, ["a", "b", "c"], "b")
    assert np.array_equal(yb, (y == 1).astype(int)) and np.allclose(P2[:, 1], P[:, 1]) and lab == ["not b", "b"]
    yb, P2, lab, mask = CMP.pairwise(y, P, ["a", "b", "c"], "c", "a")
    assert mask.sum() == np.isin(y, [0, 2]).sum() and set(yb) <= {0, 1}
    assert np.allclose(P2[:, 1], P[mask, 2] / (P[mask, 2] + P[mask, 0])) and lab == ["a", "c"]
    y2 = rng.integers(0, 2, 50)
    P3 = np.c_[1 - np.linspace(0, 1, 50), np.linspace(0, 1, 50)]
    yf, Pf, lf = CMP.flip_binary(y2, P3, ["neg", "pos"])
    assert np.array_equal(yf, 1 - y2) and np.allclose(Pf[:, 1], P3[:, 0]) and lf == ["pos", "neg"]


def test_feature_groups_split_genes_from_clinical():
    from ml_studio.core.feature_groups import split_feature_groups
    g = split_feature_groups(["age", "pack_years", "TP53", "HLA-DRB1", "C20orf27", "RP11-156E8.1", "ENSG00000141510",
                              "egfr", "tumor_size_cm"])
    assert g["omics"] == ["TP53", "HLA-DRB1", "C20orf27", "RP11-156E8.1", "ENSG00000141510"]
    assert split_feature_groups(["age", "TP53"], covariates=["TP53"])["clinical"] == ["TP53"]


def test_figure_export_formats():
    from ml_studio.visualization import export as EX
    from ml_studio.visualization.classification_plots import roc
    fig = roc(np.r_[np.zeros(20, int), np.ones(20, int)], np.c_[np.linspace(1, 0, 40), np.linspace(0, 1, 40)], ["a", "b"])
    assert EX.export(fig, "html").startswith(b"<html")
    if not EX.static_available():
        pytest.skip("no Chrome/Chromium for static export")
    from PIL import Image
    import io as _io
    im = Image.open(_io.BytesIO(EX.export(fig, "PNG", dpi=300, width_in=5)))
    assert im.size[0] == 1500 and round(im.info["dpi"][0]) == 300
    assert EX.export(fig, "svg").lstrip().startswith(b"<svg")
    assert EX.export(fig, "pdf").startswith(b"%PDF")
    assert EX.export(fig, "jpeg")[:2] == b"\xff\xd8"
