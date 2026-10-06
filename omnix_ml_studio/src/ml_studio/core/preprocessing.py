"""Leakage-safe preprocessing.

Everything that *learns* from data (missingness filter, imputation, variance and
correlation filters, power/quantile transforms, scaling, univariate selection,
resampling) lives inside one estimator pipeline. That pipeline is cloned and fitted
inside every CV training fold, so no statistic is ever computed from validation,
holdout or external data.

Log transforms are stateless but strict: values <= 0 raise an error unless the user
has explicitly chosen an offset (no silent pseudocounts).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif, f_regression
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (MinMaxScaler, OneHotEncoder, PowerTransformer,
                                   QuantileTransformer, RobustScaler, StandardScaler)
from sklearn.utils.validation import check_is_fitted

from .config import PreprocessConfig, StudyDesign
from .data_validation import infer_types


# --------------------------------------------------------------------------- data prep
@dataclass
class PreparedData:
    X: pd.DataFrame
    y: np.ndarray                 # int-encoded for classification, float for regression
    classes: Optional[list]       # original labels, index = code
    groups: Optional[np.ndarray]
    numeric: list[str]
    categorical: list[str]
    index: pd.Index               # original row index


def prepare_xy(df: pd.DataFrame, design: StudyDesign) -> PreparedData:
    """Select features, drop rows without target (explicitly) and encode the target.

    Label encoding the target is not a learned transformation of X, so it is safe to do
    once up front. For binary tasks the positive label is encoded as 1.
    """
    d = df.loc[df[design.target].notna()]
    num, cat = infer_types(d, design.features, design.categorical)
    X = d[design.features].copy()
    for c in cat:
        X[c] = X[c].astype("object").where(X[c].notna(), np.nan)
        X[c] = X[c].map(lambda v: v if pd.isna(v) else str(v))
    yraw = d[design.target]
    classes = None
    if design.task == "regression":
        y = pd.to_numeric(yraw).to_numpy(dtype=float)
    else:
        classes = sorted(yraw.astype(str).unique())
        if design.task == "binary":
            if len(classes) != 2:
                raise ValueError(f"Binary task needs exactly 2 classes, found {len(classes)}.")
            pos = str(design.positive_label) if design.positive_label is not None else classes[-1]
            if pos not in classes:
                raise ValueError(f"Positive label '{pos}' not in target values {classes}.")
            classes = [c for c in classes if c != pos] + [pos]
        y = np.array([classes.index(v) for v in yraw.astype(str)])
    groups = d[design.group].astype(str).to_numpy() if design.group else None
    return PreparedData(X, y, classes, groups, num, cat, d.index)


# --------------------------------------------------------------------------- transformers
class MissingnessFilter(TransformerMixin, BaseEstimator):
    """Drop columns whose missing fraction (in the training fold) exceeds a threshold."""

    def __init__(self, max_fraction: Optional[float] = 0.5):
        self.max_fraction = max_fraction

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        frac = X.isna().mean().to_numpy()
        keep = np.ones(X.shape[1], bool) if self.max_fraction is None else frac <= self.max_fraction
        if not keep.any():
            keep[np.argmin(frac)] = True
        self.keep_ = keep
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.missing_fraction_ = frac
        return self

    def transform(self, X):
        check_is_fitted(self, "keep_")
        X = pd.DataFrame(X, columns=self.feature_names_in_)
        return X.loc[:, self.keep_]

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_[self.keep_]


class LogTransformer(TransformerMixin, BaseEstimator):
    """log(x + offset). Raises on non-positive values when offset is None."""

    def __init__(self, base: float = 2.0, offset: Optional[float] = None):
        self.base = base
        self.offset = offset

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self._check(X)
        return self

    def _check(self, X):
        v = X.to_numpy(dtype=float)
        off = 0.0 if self.offset is None else self.offset
        bad = np.nan_to_num(v, nan=1.0) + off <= 0
        if bad.any():
            cols = list(np.asarray(X.columns)[bad.any(axis=0)][:5])
            raise ValueError(
                f"log{int(self.base)} transform: {int(bad.sum())} value(s) "
                f"{'<= 0' if self.offset is None else f'<= -{self.offset}'} in "
                f"{int(bad.any(axis=0).sum())} feature(s) (e.g. {cols}). Choose an explicit "
                "offset, a different transform, or remove these features.")

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.feature_names_in_)
        self._check(X)
        off = 0.0 if self.offset is None else self.offset
        return np.log(X.astype(float) + off) / np.log(self.base)

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_


class CorrelationFilter(TransformerMixin, BaseEstimator):
    """Greedy removal of features with |Pearson r| above threshold (fit on training fold)."""

    def __init__(self, threshold: Optional[float] = 0.95):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        p = X.shape[1]
        keep = np.ones(p, bool)
        if self.threshold is not None and p > 1:
            v = X.to_numpy(dtype=float)
            v = (v - v.mean(0)) / np.where(v.std(0) == 0, 1, v.std(0))
            C = np.abs(v.T @ v / len(v))
            np.fill_diagonal(C, 0)
            order = np.argsort(-np.nanvar(X.to_numpy(dtype=float), axis=0))  # keep higher variance
            for j in order:
                if keep[j]:
                    drop = (C[j] > self.threshold) & keep
                    drop[j] = False
                    keep[drop] = False
        self.keep_ = keep
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.feature_names_in_)
        return X.loc[:, self.keep_]

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_[self.keep_]


class SafeVarianceThreshold(VarianceThreshold):
    """VarianceThreshold that keeps at least one column (tiny folds can be all-constant)."""

    def fit(self, X, y=None):
        try:
            return super().fit(X, y)
        except ValueError:
            self.threshold = -1.0
            return super().fit(X, y)


class SafeSelectKBest(SelectKBest):
    """SelectKBest with k capped at the number of available columns."""

    def fit(self, X, y):
        k0 = self.k
        if self.k != "all" and self.k > X.shape[1]:
            self.k = "all"
        try:
            return super().fit(X, y)
        finally:
            self.k = k0

    def _get_support_mask(self):
        k0 = self.k
        if self.k != "all" and self.k > len(self.scores_):
            self.k = "all"
        try:
            return super()._get_support_mask()
        finally:
            self.k = k0


# --------------------------------------------------------------------------- builders
def check_log_validity(X: pd.DataFrame, numeric: list[str]) -> dict:
    """Report non-positive values so the user can make an explicit offset decision."""
    v = X[numeric].to_numpy(dtype=float)
    neg = np.nan_to_num(v, nan=1.0) < 0
    zero = np.nan_to_num(v, nan=1.0) == 0
    return {"n_zero": int(zero.sum()), "n_negative": int(neg.sum()),
            "features_with_nonpositive": [numeric[j] for j in np.where((neg | zero).any(0))[0]],
            "valid_without_offset": not (neg | zero).any(),
            "min_positive": float(np.nanmin(np.where(v > 0, v, np.nan))) if (v > 0).any() else None}


def _num_steps(cfg: PreprocessConfig, with_transform: bool) -> list:
    steps = []
    if cfg.max_missing_fraction is not None:
        steps.append(("missing_filter", MissingnessFilter(cfg.max_missing_fraction)))
    if with_transform and cfg.transform in ("log2", "log10"):
        steps.append(("log", LogTransformer(2.0 if cfg.transform == "log2" else 10.0, cfg.log_offset)))
    imp = {"median": SimpleImputer(strategy="median"), "mean": SimpleImputer(strategy="mean"),
           "knn": KNNImputer(n_neighbors=5),
           "constant": SimpleImputer(strategy="constant", fill_value=cfg.numeric_fill_value),
           "none": None}[cfg.numeric_imputation]
    if imp is not None:
        steps.append(("impute", imp))
    if cfg.variance_threshold is not None:
        steps.append(("variance", SafeVarianceThreshold(cfg.variance_threshold)))
    if with_transform and cfg.transform == "yeo-johnson":
        steps.append(("power", PowerTransformer(method="yeo-johnson", standardize=False)))
    elif with_transform and cfg.transform == "quantile":
        steps.append(("quantile", QuantileTransformer(output_distribution="normal",
                                                      n_quantiles=100, random_state=0)))
    sc = {"standard": StandardScaler(), "minmax": MinMaxScaler(), "robust": RobustScaler(),
          "none": None}[cfg.scaling]
    if sc is not None:
        steps.append(("scale", sc))
    if cfg.correlation_threshold is not None:
        steps.append(("corr_filter", CorrelationFilter(cfg.correlation_threshold)))
    return steps or [("passthrough", "passthrough")]


def transform_split(numeric: list[str], cfg: PreprocessConfig) -> tuple[list[str], list[str]]:
    """(columns that get cfg.transform, numeric columns left on their original scale)."""
    if cfg.transform == "none":
        return [], list(numeric)
    if cfg.transform_columns is None:
        return list(numeric), []
    keep = set(cfg.transform_columns)
    return [c for c in numeric if c in keep], [c for c in numeric if c not in keep]


def _prep_steps(numeric: list[str], categorical: list[str], cfg: PreprocessConfig,
                task: str) -> list:
    t_cols, plain_cols = transform_split(numeric, cfg)
    cat_imp = (SimpleImputer(strategy="most_frequent") if cfg.categorical_imputation == "most_frequent"
               else SimpleImputer(strategy="constant", fill_value="missing"))
    transformers = []
    if t_cols and plain_cols:          # transform only the chosen columns (e.g. omics, not age)
        transformers.append(("num", Pipeline(_num_steps(cfg, True)), t_cols))
        transformers.append(("num_plain", Pipeline(_num_steps(cfg, False)), plain_cols))
    elif numeric:
        transformers.append(("num", Pipeline(_num_steps(cfg, bool(t_cols))), numeric))
    if categorical:
        transformers.append(("cat", Pipeline([
            ("impute", cat_imp),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical))
    ct = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
    ct.set_output(transform="pandas")
    steps = [("columns", ct)]
    if cfg.univariate_k:
        score = f_regression if task == "regression" else f_classif
        sel = SafeSelectKBest(score, k=cfg.univariate_k)
        sel.set_output(transform="pandas")
        steps.append(("select", sel))
    if cfg.fs_method:
        from .feature_selection import FeatureSelector
        fs = FeatureSelector(cfg.fs_method, cfg.fs_k, task)
        fs.set_output(transform="pandas")
        steps.append(("fs", fs))
    return steps


def build_preprocessor(numeric: list[str], categorical: list[str], cfg: PreprocessConfig,
                       task: str) -> Pipeline:
    """Feature preprocessing only (no estimator), as a standalone sklearn Pipeline.
    Output is a DataFrame with traceable names: numeric unchanged, one-hot ``col_level``."""
    pre = Pipeline(_prep_steps(numeric, categorical, cfg, task))
    pre.set_output(transform="pandas")
    return pre


def make_sampler(cfg: PreprocessConfig, random_state: int = 0):
    if cfg.imbalance == "oversample":
        from imblearn.over_sampling import RandomOverSampler
        return RandomOverSampler(random_state=random_state)
    if cfg.imbalance == "undersample":
        from imblearn.under_sampling import RandomUnderSampler
        return RandomUnderSampler(random_state=random_state)
    if cfg.imbalance == "smote":
        from imblearn.over_sampling import SMOTE
        return SMOTE(random_state=random_state, k_neighbors=3)
    return None


def build_pipeline(estimator, numeric, categorical, cfg: PreprocessConfig, task: str,
                   random_state: int = 0):
    """Full pipeline: columns -> (select) -> (resampler) -> model. Flat, so that
    imblearn can run samplers during fit only.

    Uses imblearn's Pipeline so that samplers run only during ``fit`` (i.e. on training
    folds) and are skipped at predict time.
    """
    from imblearn.pipeline import Pipeline as ImbPipeline
    steps = list(_prep_steps(numeric, categorical, cfg, task))
    sampler = make_sampler(cfg, random_state) if task != "regression" else None
    if sampler is not None:
        steps.append(("resample", sampler))
    est = clone(estimator)
    if cfg.imbalance == "class_weight" and "class_weight" in est.get_params():
        est.set_params(class_weight="balanced")
    steps.append(("model", est))
    return ImbPipeline(steps)


def describe_preprocessing(cfg: PreprocessConfig) -> list[str]:
    """Human-readable, ordered description for reports."""
    out = []
    if cfg.max_missing_fraction is not None:
        out.append(f"Drop numeric features missing in > {cfg.max_missing_fraction:.0%} of training samples")
    scope = ("" if cfg.transform_columns is None
             else f" on {len(cfg.transform_columns)} selected column(s); others untransformed")
    if cfg.transform in ("log2", "log10"):
        out.append(f"{cfg.transform} transform" + (f" with explicit offset {cfg.log_offset}"
                                                   if cfg.log_offset is not None else " (no offset)") + scope)
    if cfg.numeric_imputation != "none":
        out.append(f"Numeric imputation: {cfg.numeric_imputation}")
    if cfg.variance_threshold is not None:
        out.append(f"Remove features with variance <= {cfg.variance_threshold}")
    if cfg.transform in ("yeo-johnson", "quantile"):
        out.append(f"{cfg.transform} transform" + scope)
    if cfg.scaling != "none":
        out.append(f"{cfg.scaling} scaling")
    if cfg.correlation_threshold is not None:
        out.append(f"Correlation filter |r| > {cfg.correlation_threshold}")
    out.append(f"Categorical: {cfg.categorical_imputation} imputation, one-hot encoding "
               "(unseen levels -> all zeros)")
    if cfg.univariate_k:
        out.append(f"Univariate selection: top {cfg.univariate_k} features (F-test)")
    if cfg.fs_method:
        from .feature_selection import METHODS
        out.append(f"Feature selection: {METHODS[cfg.fs_method]}, k = {cfg.fs_k}")
    if cfg.imbalance != "none":
        out.append(f"Imbalance handling: {cfg.imbalance} (training folds only)")
    out.append("All learned steps are fitted inside each training fold only.")
    return out
