import numpy as np
import pytest

from ml_studio.core import clustering as CL
from ml_studio.core import synthetic
from ml_studio.core.config import PreprocessConfig


@pytest.fixture(scope="module")
def blobs():
    df = synthetic.clusters()
    feats = [c for c in df.columns if c.startswith("f")]
    d = CL.prepare_unsupervised(df, feats, reference="true_subtype", sample_id="sample_id")
    Z = CL.build_unsupervised_preprocessor(d.numeric, d.categorical, PreprocessConfig()).fit_transform(d.X)
    return d, Z


def test_full_pycaret_cluster_list():
    assert {"kmeans", "ap", "meanshift", "sc", "hclust", "dbscan", "optics", "birch"} <= set(CL.REGISTRY)


@pytest.mark.parametrize("key", list(CL.REGISTRY))
def test_every_algorithm_runs(blobs, key):
    d, Z = blobs
    spec = CL.get(key)
    params = {"n_clusters": 3} if spec.uses_k else ({"eps": 2.5} if key == "dbscan" else {})
    model, lab = CL.fit_predict(Z, key, params)
    assert len(lab) == len(Z)
    m = CL.internal_metrics(Z, lab)
    assert m["n_clusters"] >= 1


@pytest.mark.parametrize("key", ["kmeans", "gmm", "hclust", "sc"])
def test_recovers_true_clusters(blobs, key):
    d, Z = blobs
    _, lab = CL.fit_predict(Z, key, {"n_clusters": 3})
    ext = CL.external_metrics(lab, d.reference)
    assert ext["ari"] > 0.9, ext


def test_compare_and_kscan_choose_true_k(blobs):
    d, Z = blobs
    lb, fail = CL.compare_clustering(Z, ["kmeans", "gmm", "hclust", "birch", "dbscan"], 3, d.reference)
    assert not fail and {"silhouette", "ari", "davies_bouldin"} <= set(lb.columns)
    # harder case (two of three centres close, 6 noise dims): internal indices prefer the
    # coarser k = 2, BIC of the Gaussian mixture still finds k = 3 -> criteria can disagree
    ks = CL.k_scan(Z, "kmeans", range(2, 8))
    assert ks["inertia"].is_monotonic_decreasing
    g = CL.k_scan(Z, "gmm", range(2, 6))
    assert int(g.loc[g["bic"].idxmin(), "k"]) == 3
    # well-separated case: every criterion finds k = 3
    df = synthetic.clusters(separation=6.0, noise_features=2, seed=3)
    d2 = CL.prepare_unsupervised(df, [c for c in df.columns if c.startswith("f")])
    Z2 = CL.build_unsupervised_preprocessor(d2.numeric, [], PreprocessConfig()).fit_transform(d2.X)
    ks2 = CL.k_scan(Z2, "kmeans", range(2, 8))
    assert int(ks2.loc[ks2["silhouette"].idxmax(), "k"]) == 3
    assert int(ks2.loc[ks2["calinski_harabasz"].idxmax(), "k"]) == 3
    assert int(ks2.loc[ks2["davies_bouldin"].idxmin(), "k"]) == 3


def test_stability_distinguishes_real_from_forced_structure(blobs):
    d, Z = blobs
    real = CL.bootstrap_stability(Z, "kmeans", {"n_clusters": 3}, n_boot=10)
    assert (real["per_cluster"]["mean_jaccard"] > 0.9).all() and real["ari_mean"] > 0.9
    rng = np.random.default_rng(0)
    import pandas as pd
    noise = pd.DataFrame(rng.normal(size=(300, 6)), columns=list("abcdef"))
    forced = CL.bootstrap_stability(noise, "kmeans", {"n_clusters": 6}, n_boot=10)
    assert forced["ari_mean"] < real["ari_mean"] - 0.2
    assert (forced["per_cluster"]["assessment"] != "stable").any()


def test_profiles_and_assignment(blobs):
    d, Z = blobs
    model, lab = CL.fit_predict(Z, "kmeans", {"n_clusters": 3})
    prof = CL.cluster_profiles(d.X, lab, d.numeric, d.categorical)
    top = set(prof["feature"].head(6))
    assert top <= {f"f{i:04d}" for i in range(1, 7)}            # informative dims first
    new, how = CL.assign_new(model, "kmeans", Z, lab, Z.head(10))
    assert how == "native predict" and (new == lab[:10]).all()
    m2, lab2 = CL.fit_predict(Z, "hclust", {"n_clusters": 3})
    new2, how2 = CL.assign_new(m2, "hclust", Z, lab2, Z.head(10))
    assert "approximation" in how2 and (new2 == lab2[:10]).all()


def test_mixed_types_and_pca(binary_df):
    feats = [c for c in binary_df.columns if c not in ("sample_id", "outcome")]
    d = CL.prepare_unsupervised(binary_df, feats)
    Z = CL.build_unsupervised_preprocessor(d.numeric, d.categorical, PreprocessConfig(), pca_components=5) \
        .fit_transform(d.X)
    assert Z.shape == (len(binary_df), 5)
    _, lab = CL.fit_predict(Z, "kmeans", {"n_clusters": 3})
    assert set(lab) == {0, 1, 2}
    e, ev = CL.embedding_2d(Z)
    assert e.shape == (len(Z), 2) and ev.sum() <= 1
