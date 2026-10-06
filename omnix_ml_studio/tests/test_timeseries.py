import numpy as np
import pandas as pd
import pytest

from ml_studio.core import synthetic
from ml_studio.core import timeseries as TS


@pytest.fixture(scope="module")
def series():
    return TS.prepare_series(synthetic.monthly_series(), "value", "date")


def test_full_pycaret_forecaster_list():
    core = {"naive", "grand_means", "snaive", "polytrend", "arima", "auto_arima", "exp_smooth", "ets", "theta",
            "stlf", "croston", "bats", "tbats"}
    red = {f"{k}_cds_dt" for k in ("lr", "en", "ridge", "lasso", "llar", "br", "huber", "omp", "par", "knn", "dt",
                                   "rf", "et", "gbr", "ada", "lightgbm", "xgboost", "catboost")}
    assert core | red <= set(TS.REGISTRY), (core | red) - set(TS.REGISTRY)


def test_prepare_infers_frequency_and_season(series):
    assert series.freq in ("MS", "M") and series.sp_suggested == 12
    assert isinstance(series.y.index, pd.PeriodIndex) and len(series.y) == 144


def test_irregular_and_missing_handled():
    df = synthetic.monthly_series(n=60).drop(index=[10, 11, 30]).reset_index(drop=True)
    df.loc[5, "value"] = np.nan
    s = TS.prepare_series(df, "value", "date")
    assert len(s.y) == 60 and not s.y.isna().any()
    assert any("missing period" in i for i in s.issues) and s.n_imputed == 4


def test_split_is_temporal_and_sealed(series):
    d = TS.TSDesign(fh=12, sp=12, n_folds=3)
    n_dev, folds = d.split(len(series.y))
    assert n_dev == 132 and folds == [(96, 108), (108, 120), (120, 132)]
    assert all(b <= n_dev for _, b in folds)              # holdout never used in CV
    assert TS.TSDesign(fh=40, sp=12, n_folds=3).validate(144)


@pytest.mark.parametrize("key", [k for k in TS.REGISTRY if TS.get(k).speed != "slow"])
def test_every_fast_forecaster_runs(series, key):
    d = TS.TSDesign(fh=12, sp=12, n_folds=2)
    cv = TS.cross_validate_forecaster(series.y, key, d)
    assert len(cv["folds"]) == 2 and np.isfinite(cv["folds"]["mae"]).all()


def test_slow_forecasters_run_once(series):
    for key in ("auto_arima", "bats", "tbats"):
        r = TS.evaluate_holdout(series.y.iloc[-72:], key, TS.TSDesign(fh=12, sp=12, n_folds=1))
        assert np.isfinite(r["metrics"]["mae"]), key


def test_seasonal_models_beat_naive_and_mase_reference(series):
    d = TS.TSDesign(fh=12, sp=12, n_folds=3)
    lb, fail, _ = TS.compare_forecasters(series.y, ["naive", "snaive", "exp_smooth", "theta", "ridge_cds_dt"], d)
    assert not fail
    lb = lb.set_index("key")
    assert lb.loc["exp_smooth", "mase_mean"] < lb.loc["snaive", "mase_mean"] < lb.loc["naive", "mase_mean"]
    assert lb.loc["exp_smooth", "beats_seasonal_naive"]


def test_mase_definition():
    ytr = np.array([1, 2, 3, 4, 5, 6.0])
    m = TS.ts_metrics(ytr, [7, 8], [7, 8], 1)
    assert m["mase"] == 0 and m["mae"] == 0
    m = TS.ts_metrics(ytr, [7, 8], [8, 9], 1)            # naive in-sample MAE = 1
    assert m["mase"] == pytest.approx(1.0) and m["rmsse"] == pytest.approx(1.0)


def test_holdout_intervals_and_final_forecast(series):
    d = TS.TSDesign(fh=12, sp=12, n_folds=2, coverage=0.9)
    assert not TS.supports_intervals("exp_smooth") and TS.supports_intervals("theta")
    assert "coverage" not in TS.evaluate_holdout(series.y, "exp_smooth", d)["metrics"]
    r = TS.evaluate_holdout(series.y, "theta", d)
    assert {"lower", "upper"} <= set(r["table"].columns) and 0.5 <= r["metrics"]["coverage"] <= 1
    assert r["table"]["time"].iloc[0] == str(series.y.index[132])
    f = TS.final_forecast(series.y, "theta", d, horizon=6)
    assert len(f["table"]) == 6 and f["table"]["time"].iloc[0] == str(series.y.index[-1] + 1)


def test_diagnostics(series):
    dg = TS.diagnostics(series.y, 12)
    assert dg["seasonal_strength"] > 0.6 and dg["trend_strength"] > 0.8
    assert "stl" in dg and len(dg["acf"]) > 12
    rng = np.random.default_rng(0)
    assert TS.ljung_box(rng.normal(size=200)) > 0.05
    assert TS.ljung_box(np.sin(np.arange(200) / 3)) < 0.01


def test_no_date_column_and_short_series():
    s = TS.prepare_series(pd.DataFrame({"v": np.sin(np.arange(80) * 2 * np.pi / 7) + 5}), "v")
    assert isinstance(s.y.index, pd.RangeIndex) and s.sp_suggested == 7
    with pytest.raises(ValueError):
        TS.prepare_series(pd.DataFrame({"v": list("abc")}), "v")
