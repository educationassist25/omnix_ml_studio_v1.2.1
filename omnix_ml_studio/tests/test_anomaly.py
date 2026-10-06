import numpy as np
import pytest

from ml_studio.core import anomaly as AN
from ml_studio.core import clustering as CL
from ml_studio.core import synthetic
from ml_studio.core.config import PreprocessConfig


@pytest.fixture(scope="module")
def data():
    df = synthetic.anomalies()
    feats = [c for c in df.columns if c.startswith("f")]
    d = CL.prepare_unsupervised(df, feats, reference="is_anomaly", sample_id="sample_id")
    Z = CL.build_unsupervised_preprocessor(d.numeric, [], PreprocessConfig(scaling="robust")).fit_transform(d.X)
    return df, d, Z


def test_full_pycaret_anomaly_list():
    assert {"abod", "cluster", "cof", "iforest", "histogram", "knn", "lof", "svm", "pca", "mcd", "sod",
            "sos"} <= set(AN.REGISTRY)


@pytest.mark.parametrize("key", list(AN.REGISTRY))
def test_every_detector_runs_and_beats_chance(data, key):
    df, d, Z = data
    model, s, f = AN.fit_detector(Z, key, 0.05)
    assert len(s) == len(Z) and f.sum() > 0
    m = AN.label_metrics(s, f, d.reference)
    assert m["roc_auc"] > 0.7, (key, m)
    if AN.get(key).transductive:
        with pytest.raises(AN.TransductiveError):
            AN.score_new(model, Z.head(20), key)
    else:
        s2, f2 = AN.score_new(model, Z.head(20), key)
        assert len(s2) == 20 and set(np.unique(f2)) <= {0, 1}
        # inductive: a sample's score does not depend on which other samples are in the batch
        alone = AN.score_new(model, Z.iloc[[3]], key)[0][0]
        assert np.isclose(alone, s2[3], rtol=1e-6, atol=1e-9), (key, alone, s2[3])


def test_strong_detectors_find_injected_anomalies(data):
    df, d, Z = data
    for key in ("iforest", "knn", "mcd"):
        _, s, f = AN.fit_detector(Z, key, 0.05)
        m = AN.label_metrics(s, f, d.reference)
        assert m["roc_auc"] > 0.9 and m["precision_at_n"] > 0.6, (key, m)


def test_compare_agreement_consensus(data):
    df, d, Z = data
    r = AN.compare_detectors(Z, ["iforest", "knn", "lof", "histogram", "mcd"], 0.05, d.reference)
    assert not r.failures and len(r.table) == 5
    assert r.agreement.shape == (5, 5) and np.allclose(np.diag(r.agreement), 1)
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(d.reference, r.consensus) > 0.9


def test_flag_stability_and_deviation(data):
    df, d, Z = data
    st = AN.flag_stability(Z, "iforest", 0.05, n_boot=8)
    assert 0 <= st["mean_jaccard"] <= 1 and st["stable_flags"] > 0
    true_idx = df.index[df["is_anomaly"] == 1][0]
    prof = AN.deviation_profile(d.X, d.numeric, true_idx)
    assert prof["abs_z"].iloc[0] > 3        # the shifted features stand out


def test_no_reference_mode(data):
    df, d, Z = data
    r = AN.compare_detectors(Z, ["iforest", "lof"], 0.1, None)
    assert "roc_auc" not in r.table.columns and (r.table["n_flagged"] > 0).all()


def test_transductive_stability_runs(data):
    df, d, Z = data
    st = AN.flag_stability(Z.iloc[:200], "cof", 0.05, n_boot=3)
    assert 0 <= st["mean_jaccard"] <= 1
