import numpy as np
import pandas as pd
import pytest
from conftest import design_for

from ml_studio.core.config import PreprocessConfig
from ml_studio.core.preprocessing import (CorrelationFilter, LogTransformer, MissingnessFilter,
                                          build_preprocessor, check_log_validity, prepare_xy)


def test_prepare_xy_encodes_positive_label_last(binary_df):
    d = prepare_xy(binary_df, design_for(binary_df, "outcome", "binary", positive_label="case"))
    assert d.classes == ["control", "case"]
    assert set(d.y) == {0, 1}
    assert (d.y == 1).sum() == (binary_df["outcome"] == "case").sum()
    assert d.categorical == ["sex", "stage"] and len(d.numeric) == 20


def test_log_transform_refuses_nonpositive_without_offset():
    X = pd.DataFrame({"a": [1.0, 2, 0], "b": [4.0, 8, 16]})
    with pytest.raises(ValueError, match="explicit"):
        LogTransformer(2).fit(X)
    out = LogTransformer(2, offset=1).fit_transform(X)
    assert np.allclose(out["b"], np.log2([5, 9, 17]))
    assert check_log_validity(X, ["a", "b"])["n_zero"] == 1
    # transform-time check: valid in train, invalid in new data
    lt = LogTransformer(10).fit(pd.DataFrame({"a": [1.0, 10]}))
    with pytest.raises(ValueError):
        lt.transform(pd.DataFrame({"a": [-1.0]}))


def test_missingness_and_correlation_filters():
    X = pd.DataFrame({"a": [1, np.nan, np.nan, np.nan], "b": [1, 2, 3, 4.0], "c": [2, 4, 6, 8.1]})
    mf = MissingnessFilter(0.5).fit(X)
    assert list(mf.get_feature_names_out()) == ["b", "c"]
    cf = CorrelationFilter(0.95).fit(X[["b", "c"]])
    assert cf.keep_.sum() == 1


@pytest.mark.parametrize("transform", ["none", "yeo-johnson", "quantile"])
@pytest.mark.parametrize("scaling", ["standard", "minmax", "robust", "none"])
@pytest.mark.parametrize("imp", ["median", "mean", "knn", "constant"])
def test_preprocessor_combinations_run(binary_df, transform, scaling, imp):
    d = prepare_xy(binary_df, design_for(binary_df, "outcome", "binary"))
    cfg = PreprocessConfig(numeric_imputation=imp, transform=transform, scaling=scaling,
                           correlation_threshold=0.9, univariate_k=10)
    pre = build_preprocessor(d.numeric, d.categorical, cfg, "binary")
    Z = pre.fit_transform(d.X, d.y)
    assert Z.shape == (300, 10)
    assert not Z.isna().any().any()
    assert all(isinstance(c, str) for c in Z.columns)


def test_feature_names_preserved(binary_df):
    d = prepare_xy(binary_df, design_for(binary_df, "outcome", "binary"))
    Z = build_preprocessor(d.numeric, d.categorical, PreprocessConfig(), "binary").fit_transform(d.X)
    assert "f0001" in Z.columns and "stage_III" in Z.columns and "sex_M" in Z.columns


def test_unseen_category_does_not_crash(binary_df):
    d = prepare_xy(binary_df, design_for(binary_df, "outcome", "binary"))
    pre = build_preprocessor(d.numeric, d.categorical, PreprocessConfig(), "binary").fit(d.X)
    new = d.X.head(3).copy(); new["stage"] = "IV"
    Z = pre.transform(new)
    assert (Z[[c for c in Z.columns if c.startswith("stage_")]].sum(axis=1) == 0).all()


def test_log_pipeline_on_omics(omics_df):
    d = prepare_xy(omics_df, design_for(omics_df, "group", "binary"))
    with pytest.raises(ValueError):
        build_preprocessor(d.numeric, [], PreprocessConfig(transform="log2"), "binary").fit(d.X)
    pre = build_preprocessor(d.numeric, [], PreprocessConfig(transform="log2", log_offset=1.0,
                                                             univariate_k=50), "binary")
    assert pre.fit_transform(d.X, d.y).shape == (80, 50)


def test_transform_only_selected_columns():
    """Regression: log on all numeric failed when clinical columns contain zeros (pack_years)."""
    import pandas as pd
    df = pd.DataFrame({"pack_years": [0.0, 10, 0, 30, 5, 0] * 10, "age": [60.0, 70, 55, 80, 65, 50] * 10,
                       "GENE1": [5.0, 50, 500, 8, 80, 800] * 10, "y": [0, 1, 0, 1, 0, 1] * 10})
    from ml_studio.core.config import StudyDesign
    d = prepare_xy(df, StudyDesign(target="y", task="binary", features=["pack_years", "age", "GENE1"]))
    with pytest.raises(ValueError, match="explicit"):
        build_preprocessor(d.numeric, [], PreprocessConfig(transform="log2", scaling="none"), "binary").fit(d.X)
    pre = build_preprocessor(d.numeric, [], PreprocessConfig(transform="log2", scaling="none",
                                                             transform_columns=["GENE1"]), "binary")
    Z = pre.fit_transform(d.X)
    assert np.allclose(Z["GENE1"], np.log2(df["GENE1"]))
    assert np.allclose(Z["pack_years"], df["pack_years"]) and np.allclose(Z["age"], df["age"])
    assert set(Z.columns) == {"pack_years", "age", "GENE1"}
