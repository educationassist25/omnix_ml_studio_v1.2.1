"""Clustering: PyCaret's algorithm list (same IDs) + Gaussian mixture.

Evaluation separates three questions:

* internal quality (silhouette, Calinski–Harabasz, Davies–Bouldin) — computed on the
  model-input space;
* agreement with an optional reference label (ARI, AMI, homogeneity, completeness,
  V-measure) — the label is never used for fitting;
* stability — refit on bootstrap subsamples and measure how well each cluster is
  recovered (Hennig's clusterwise Jaccard) and overall agreement (ARI).

Cluster profiles report univariate tests between clusters, flagged as descriptive: the
clusters were derived from the same features, so those p-values are optimistic
(post-selection inference) and must not be read as confirmatory evidence.
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd
from sklearn import metrics as M
from sklearn.cluster import (DBSCAN, OPTICS, AffinityPropagation, AgglomerativeClustering, Birch, KMeans,
                             MeanShift, SpectralClustering, estimate_bandwidth)
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from .config import PreprocessConfig
from .data_validation import infer_types


@dataclass
class ClusterSpec:
    key: str
    name: str
    factory: Callable[[int, dict], object]
    params: dict = field(default_factory=dict)       # tunable params: name -> (default, kind, lo, hi)
    uses_k: bool = True
    can_predict: bool = True
    noise: bool = False                              # labels -1 as noise
    family: str = "Centroid"
    notes: str = ""
    pycaret: bool = True
    speed: str = "fast"


def _kmodes(rs, p):
    from kmodes.kmodes import KModes
    return KModes(n_clusters=p.get("n_clusters", 4), random_state=rs, n_init=5)


REGISTRY: dict[str, ClusterSpec] = {s.key: s for s in [
    ClusterSpec("kmeans", "K-Means", lambda rs, p: KMeans(p.get("n_clusters", 4), n_init=10, random_state=rs),
                {"n_clusters": (4, "int", 2, 20)}, family="Centroid"),
    ClusterSpec("gmm", "Gaussian Mixture", lambda rs, p: GaussianMixture(p.get("n_clusters", 4),
                                                                         covariance_type=p.get("covariance", "full"),
                                                                         random_state=rs, n_init=3),
                {"n_clusters": (4, "int", 2, 20), "covariance": ("full", "choice", ["full", "diag", "tied", "spherical"])},
                family="Model-based", pycaret=False, notes="Soft assignments; choose k by BIC."),
    ClusterSpec("hclust", "Agglomerative (hierarchical)",
                lambda rs, p: AgglomerativeClustering(p.get("n_clusters", 4), linkage=p.get("linkage", "ward")),
                {"n_clusters": (4, "int", 2, 20), "linkage": ("ward", "choice", ["ward", "average", "complete", "single"])},
                can_predict=False, family="Hierarchical"),
    ClusterSpec("birch", "BIRCH", lambda rs, p: Birch(n_clusters=p.get("n_clusters", 4),
                                                      threshold=p.get("threshold", 0.5)),
                {"n_clusters": (4, "int", 2, 20), "threshold": (0.5, "float", 0.05, 3.0)}, family="Hierarchical"),
    ClusterSpec("sc", "Spectral", lambda rs, p: SpectralClustering(p.get("n_clusters", 4), random_state=rs,
                                                                   affinity="nearest_neighbors",
                                                                   n_neighbors=p.get("n_neighbors", 10)),
                {"n_clusters": (4, "int", 2, 20), "n_neighbors": (10, "int", 3, 50)}, can_predict=False,
                family="Graph", speed="medium"),
    ClusterSpec("dbscan", "DBSCAN", lambda rs, p: DBSCAN(eps=p.get("eps", 0.5), min_samples=p.get("min_samples", 5)),
                {"eps": (0.5, "float", 0.05, 10.0), "min_samples": (5, "int", 2, 50)}, uses_k=False,
                can_predict=False, noise=True, family="Density",
                notes="Finds the number of clusters itself; points in sparse regions are labelled noise."),
    ClusterSpec("optics", "OPTICS", lambda rs, p: OPTICS(min_samples=p.get("min_samples", 5)),
                {"min_samples": (5, "int", 2, 50)}, uses_k=False, can_predict=False, noise=True, family="Density",
                speed="medium"),
    ClusterSpec("meanshift", "Mean Shift",
                lambda rs, p: MeanShift(bandwidth=p.get("bandwidth"), bin_seeding=True),
                {"bandwidth": (None, "float", 0.1, 20.0)}, uses_k=False, family="Density", speed="medium",
                notes="Bandwidth estimated from the data when left empty."),
    ClusterSpec("ap", "Affinity Propagation",
                lambda rs, p: AffinityPropagation(damping=p.get("damping", 0.7), random_state=rs, max_iter=500),
                {"damping": (0.7, "float", 0.5, 0.99)}, uses_k=False, family="Exemplar", speed="medium",
                notes="Often returns many clusters; cost grows with n²."),
]}
if importlib.util.find_spec("kmodes") is not None:
    REGISTRY["kmodes"] = ClusterSpec("kmodes", "K-Modes (categorical)", _kmodes, {"n_clusters": (4, "int", 2, 20)},
                                     family="Centroid", notes="For categorical features only.")

DEFAULT = ["kmeans", "gmm", "hclust", "birch", "sc", "dbscan"]


def get(key: str) -> ClusterSpec:
    if key not in REGISTRY:
        raise KeyError(f"Unknown clustering algorithm '{key}'. Available: {list(REGISTRY)}")
    return REGISTRY[key]


# --------------------------------------------------------------------------- data
@dataclass
class UnsupervisedData:
    X: pd.DataFrame
    numeric: list
    categorical: list
    reference: Optional[np.ndarray]      # optional true labels / anomaly flags (evaluation only)
    reference_name: Optional[str]
    ids: Optional[pd.Series]
    index: pd.Index


def prepare_unsupervised(df: pd.DataFrame, features: list, reference: Optional[str] = None,
                         sample_id: Optional[str] = None, categorical_override=None) -> UnsupervisedData:
    num, cat = infer_types(df, features, categorical_override)
    X = df[features].copy()
    for c in cat:
        X[c] = X[c].map(lambda v: v if pd.isna(v) else str(v))
    ref = df[reference].to_numpy() if reference else None
    return UnsupervisedData(X, num, cat, ref, reference, df[sample_id] if sample_id else None, df.index)


def build_unsupervised_preprocessor(numeric, categorical, cfg: PreprocessConfig,
                                    pca_components: Optional[int] = None):
    """Imputation / transform / scaling / one-hot (no target-based steps), optional PCA."""
    from dataclasses import replace

    from .preprocessing import _prep_steps
    cfg = replace(cfg, univariate_k=None, fs_method=None, imbalance="none")
    steps = _prep_steps(numeric, categorical, cfg, "regression")
    if pca_components:
        pca = PCA(n_components=pca_components, random_state=0)
        pca.set_output(transform="pandas")
        steps.append(("pca", pca))
    pipe = Pipeline(steps)
    pipe.set_output(transform="pandas")
    return pipe


# --------------------------------------------------------------------------- fitting & metrics
def fit_predict(Z: pd.DataFrame, key: str, params: Optional[dict] = None, random_state: int = 0):
    spec = get(key)
    p = dict(params or {})
    if key == "meanshift" and not p.get("bandwidth"):
        p["bandwidth"] = estimate_bandwidth(Z.to_numpy(float), quantile=0.2, n_samples=min(500, len(Z)),
                                            random_state=random_state) or 1.0
    model = spec.factory(random_state, p)
    labels = np.asarray(model.fit_predict(Z.to_numpy(float)))
    return model, labels


def internal_metrics(Z: pd.DataFrame, labels: np.ndarray) -> dict:
    v = Z.to_numpy(float)
    keep = labels != -1
    n_cl = len(set(labels[keep]))
    out = {"n_clusters": n_cl, "noise_fraction": float((~keep).mean()),
           "largest_cluster_fraction": float(pd.Series(labels[keep]).value_counts(normalize=True).max()) if keep.any() else np.nan}
    if n_cl >= 2 and keep.sum() > n_cl:
        out["silhouette"] = float(M.silhouette_score(v[keep], labels[keep], sample_size=min(3000, int(keep.sum())),
                                                     random_state=0))
        out["calinski_harabasz"] = float(M.calinski_harabasz_score(v[keep], labels[keep]))
        out["davies_bouldin"] = float(M.davies_bouldin_score(v[keep], labels[keep]))
    else:
        out.update(silhouette=np.nan, calinski_harabasz=np.nan, davies_bouldin=np.nan)
    return out


def external_metrics(labels: np.ndarray, reference: np.ndarray) -> dict:
    ok = pd.notna(reference)
    r, l_ = pd.Series(reference[ok]).astype(str).to_numpy(), labels[ok]
    return {"ari": M.adjusted_rand_score(r, l_), "ami": M.adjusted_mutual_info_score(r, l_),
            "homogeneity": M.homogeneity_score(r, l_), "completeness": M.completeness_score(r, l_),
            "v_measure": M.v_measure_score(r, l_)}


CLUSTER_METRIC_LABELS = {"n_clusters": "Clusters", "noise_fraction": "Noise fraction",
                         "largest_cluster_fraction": "Largest cluster", "silhouette": "Silhouette",
                         "calinski_harabasz": "Calinski–Harabasz", "davies_bouldin": "Davies–Bouldin",
                         "ari": "ARI (vs reference)", "ami": "AMI (vs reference)", "homogeneity": "Homogeneity",
                         "completeness": "Completeness", "v_measure": "V-measure"}


def compare_clustering(Z: pd.DataFrame, keys: list, n_clusters: int = 4, reference=None, random_state: int = 0,
                       progress=None) -> tuple[pd.DataFrame, dict]:
    import time
    rows, failures = [], {}
    for i, k in enumerate(keys):
        spec = get(k)
        if progress:
            progress(i, len(keys), spec.name)
        t = time.time()
        try:
            _, lab = fit_predict(Z, k, {"n_clusters": n_clusters} if spec.uses_k else {}, random_state)
        except Exception as e:  # noqa: BLE001
            failures[k] = f"{type(e).__name__}: {e}"
            continue
        row = {"key": k, "Model": spec.name, "family": spec.family, **internal_metrics(Z, lab),
               "fit_time_s": time.time() - t}
        if reference is not None:
            row.update(external_metrics(lab, reference))
        rows.append(row)
    lb = pd.DataFrame(rows)
    if len(lb):
        lb = lb.sort_values("silhouette", ascending=False, na_position="last").reset_index(drop=True)
    return lb, failures


def k_scan(Z: pd.DataFrame, key: str, k_range=range(2, 11), random_state: int = 0, extra: Optional[dict] = None):
    """Internal metrics (and inertia / BIC where defined) across k."""
    rows = []
    for k in k_range:
        p = {"n_clusters": k, **(extra or {})}
        model, lab = fit_predict(Z, key, p, random_state)
        r = {"k": k, **internal_metrics(Z, lab)}
        if hasattr(model, "inertia_"):
            r["inertia"] = float(model.inertia_)
        if isinstance(model, GaussianMixture):
            r["bic"] = float(model.bic(Z.to_numpy(float)))
        rows.append(r)
    return pd.DataFrame(rows)


def bootstrap_stability(Z: pd.DataFrame, key: str, params: dict, n_boot: int = 20, frac: float = 0.8,
                        random_state: int = 0) -> dict:
    """Hennig-style clusterwise Jaccard + ARI between full-data and subsample clusterings
    (evaluated on the subsample members). Jaccard > 0.75 ≈ stable, < 0.5 ≈ dissolved."""
    _, ref = fit_predict(Z, key, params, random_state)
    rng = np.random.default_rng(random_state)
    n = len(Z)
    clusters = [c for c in np.unique(ref) if c != -1]
    jac = {c: [] for c in clusters}
    aris = []
    for b in range(n_boot):
        ii = np.sort(rng.choice(n, int(frac * n), replace=False))
        _, lab = fit_predict(Z.iloc[ii], key, params, random_state + b + 1)
        r = ref[ii]
        aris.append(M.adjusted_rand_score(r, lab))
        for c in clusters:
            A = set(np.where(r == c)[0])
            if not A:
                continue
            best = 0.0
            for c2 in set(lab) - {-1}:
                B = set(np.where(lab == c2)[0])
                best = max(best, len(A & B) / len(A | B))
            jac[c].append(best)
    per = pd.DataFrame({"cluster": clusters, "size": [int((ref == c).sum()) for c in clusters],
                        "mean_jaccard": [np.mean(jac[c]) if jac[c] else np.nan for c in clusters],
                        "sd_jaccard": [np.std(jac[c], ddof=1) if len(jac[c]) > 1 else np.nan for c in clusters]})
    per["assessment"] = np.select([per["mean_jaccard"] >= 0.75, per["mean_jaccard"] >= 0.6,
                                   per["mean_jaccard"] >= 0.5], ["stable", "pattern", "weak"], "dissolved")
    return {"per_cluster": per, "ari_mean": float(np.mean(aris)), "ari_sd": float(np.std(aris, ddof=1)),
            "n_boot": n_boot, "fraction": frac, "labels": ref}


def cluster_profiles(X: pd.DataFrame, labels: np.ndarray, numeric: list, categorical: list) -> pd.DataFrame:
    """Per-feature descriptive differences between clusters (Kruskal–Wallis / χ², BH-FDR).
    Descriptive only: clusters were built from these features (post-selection bias)."""
    from scipy import stats
    from statsmodels.stats.multitest import multipletests
    keep = labels != -1
    Xk, lk = X.loc[keep], labels[keep]
    rows = []
    for c in numeric:
        x = Xk[c].to_numpy(float)
        groups = [x[(lk == g) & np.isfinite(x)] for g in np.unique(lk)]
        groups = [g for g in groups if len(g) > 0]
        try:
            h, p = stats.kruskal(*groups) if len(groups) > 1 else (np.nan, 1.0)
        except ValueError:
            h, p = np.nan, 1.0
        n = sum(len(g) for g in groups)
        eff = (h - len(groups) + 1) / (n - len(groups)) if np.isfinite(h) and n > len(groups) else np.nan
        rows.append({"feature": c, "type": "numeric", "effect (ε²)": eff, "p": p,
                     **{f"median c{g}": float(np.nanmedian(x[lk == g])) for g in np.unique(lk)}})
    for c in categorical:
        ct = pd.crosstab(Xk[c].astype(str), lk)
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            continue
        chi2, p = stats.chi2_contingency(ct, correction=False)[:2]
        v = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
        rows.append({"feature": c, "type": "categorical", "effect (ε²)": v, "p": p,
                     **{f"median c{g}": ct[g].idxmax() for g in ct.columns}})
    out = pd.DataFrame(rows)
    if len(out):
        out["q_bh"] = multipletests(out["p"].fillna(1.0), method="fdr_bh")[1]
        out = out.sort_values("effect (ε²)", ascending=False).reset_index(drop=True)
    return out


def assign_new(model, key: str, Z_train: pd.DataFrame, train_labels: np.ndarray, Z_new: pd.DataFrame):
    """Cluster labels for new samples. Algorithms without ``predict`` use a 1-nearest-neighbour
    rule on the training assignment (an approximation, reported as such)."""
    spec = get(key)
    if spec.can_predict and hasattr(model, "predict"):
        return np.asarray(model.predict(Z_new.to_numpy(float))), "native predict"
    keep = train_labels != -1
    knn = KNeighborsClassifier(n_neighbors=1).fit(Z_train.to_numpy(float)[keep], train_labels[keep])
    return knn.predict(Z_new.to_numpy(float)), "nearest training sample (approximation)"


def embedding_2d(Z: pd.DataFrame, method: str = "pca", random_state: int = 0):
    v = Z.to_numpy(float)
    if method == "tsne":
        from sklearn.manifold import TSNE
        return TSNE(2, random_state=random_state, perplexity=min(30, max(5, len(v) // 10)),
                    init="pca").fit_transform(v), None
    p = PCA(2, random_state=random_state).fit(v)
    return p.transform(v), p.explained_variance_ratio_
