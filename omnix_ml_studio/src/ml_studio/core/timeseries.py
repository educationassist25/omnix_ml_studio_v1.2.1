"""Univariate time-series forecasting: PyCaret's forecaster list (same IDs) via sktime.

Validation respects time order:
* holdout = the last ``fh`` observations, sealed before model development;
* rolling-origin CV on the development part: expanding training window, ``n_folds`` origins
  each forecasting the next ``fh`` points, model refit at every origin;
* metrics: MASE and RMSSE (scaled by the in-sample seasonal-naive error, so 1.0 = as good
  as seasonal naive), MAE, RMSE, MAPE (only if no zeros), sMAPE;
* prediction-interval coverage on the holdout when the model supports intervals.
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

FREQ_SP = {"M": 12, "MS": 12, "Q": 4, "QS": 4, "W": 52, "D": 7, "B": 5, "H": 24, "h": 24, "A": 1, "Y": 1,
           "YS": 1, "AS": 1, "T": 60, "min": 60}


@dataclass
class ForecasterSpec:
    key: str
    name: str
    factory: Callable[[int, int, dict], object]       # (sp, random_state, params)
    family: str = "Statistical"
    speed: str = "fast"
    notes: str = ""
    params: dict = field(default_factory=dict)


def _reduction(make_reg, wl_default=None):
    def f(sp, rs, p):
        from sktime.forecasting.compose import TransformedTargetForecaster, make_reduction
        from sktime.forecasting.trend import PolynomialTrendForecaster
        from sktime.transformations.series.detrend import ConditionalDeseasonalizer, Detrender
        wl = int(p.get("window_length") or wl_default or max(sp, 3))
        steps = []
        if sp > 1:
            steps.append(("deseasonalize", ConditionalDeseasonalizer(sp=sp)))
        steps += [("detrend", Detrender(PolynomialTrendForecaster(degree=1))),
                  ("forecast", make_reduction(make_reg(rs), window_length=wl, strategy="recursive"))]
        return TransformedTargetForecaster(steps)
    return f


def _sk(path, **kw):
    mod, cls = path.rsplit(".", 1)

    def f(sp, rs, p):
        import importlib
        C = getattr(importlib.import_module(mod), cls)
        args = {k: (v(sp) if callable(v) else v) for k, v in kw.items()}
        args.update({k: v for k, v in p.items() if v is not None})
        return C(**args)
    return f


def _reg(name):
    def make(rs):
        import sklearn.ensemble as E
        import sklearn.linear_model as L
        import sklearn.neighbors as N
        import sklearn.tree as T
        m = {"lr": lambda: L.LinearRegression(), "en": lambda: L.ElasticNet(alpha=0.1, random_state=rs),
             "ridge": lambda: L.Ridge(), "lasso": lambda: L.Lasso(alpha=0.1, random_state=rs),
             "llar": lambda: L.LassoLars(alpha=0.1, random_state=rs), "br": lambda: L.BayesianRidge(),
             "huber": lambda: L.HuberRegressor(max_iter=1000), "omp": lambda: L.OrthogonalMatchingPursuit(),
             "par": lambda: L.PassiveAggressiveRegressor(random_state=rs),
             "knn": lambda: N.KNeighborsRegressor(), "dt": lambda: T.DecisionTreeRegressor(random_state=rs),
             "rf": lambda: E.RandomForestRegressor(200, random_state=rs, n_jobs=1),
             "et": lambda: E.ExtraTreesRegressor(200, random_state=rs, n_jobs=1),
             "gbr": lambda: E.GradientBoostingRegressor(random_state=rs),
             "ada": lambda: E.AdaBoostRegressor(random_state=rs)}
        if name == "lightgbm":
            from lightgbm import LGBMRegressor
            return LGBMRegressor(random_state=rs, verbose=-1, n_jobs=1, min_child_samples=5)
        if name == "xgboost":
            from xgboost import XGBRegressor
            return XGBRegressor(random_state=rs, n_jobs=1, verbosity=0)
        if name == "catboost":
            from catboost import CatBoostRegressor
            return CatBoostRegressor(random_state=rs, verbose=0, thread_count=1, iterations=200,
                                     allow_writing_files=False)
        return m[name]()
    return make


_SEAS = lambda sp: sp if sp > 1 else 1  # noqa: E731
REGISTRY: dict[str, ForecasterSpec] = {s.key: s for s in [
    ForecasterSpec("naive", "Naive (last value)", _sk("sktime.forecasting.naive.NaiveForecaster", strategy="last"),
                   family="Baseline"),
    ForecasterSpec("grand_means", "Grand mean", _sk("sktime.forecasting.naive.NaiveForecaster", strategy="mean"),
                   family="Baseline"),
    ForecasterSpec("snaive", "Seasonal naive", _sk("sktime.forecasting.naive.NaiveForecaster", strategy="last", sp=_SEAS),
                   family="Baseline", notes="Reference for MASE / RMSSE."),
    ForecasterSpec("polytrend", "Polynomial trend", _sk("sktime.forecasting.trend.PolynomialTrendForecaster", degree=1),
                   family="Baseline", params={"degree": (1, "int", 1, 3)}),
    ForecasterSpec("arima", "ARIMA", lambda sp, rs, p: __import__("sktime.forecasting.arima", fromlist=["ARIMA"]).ARIMA(
        order=tuple(p.get("order", (1, 0, 0))), seasonal_order=(0, 1, 0, sp) if sp > 1 else (0, 0, 0, 0),
        suppress_warnings=True), family="Statistical"),
    ForecasterSpec("auto_arima", "Auto ARIMA", _sk("sktime.forecasting.arima.AutoARIMA", sp=_SEAS, suppress_warnings=True,
                                                    seasonal=True, max_p=3, max_q=3, n_jobs=1),
                   family="Statistical", speed="slow", notes="Stepwise search over (p, d, q)(P, D, Q)."),
    ForecasterSpec("exp_smooth", "Exponential smoothing (Holt–Winters)",
                   lambda sp, rs, p: __import__("sktime.forecasting.exp_smoothing", fromlist=["E"]).ExponentialSmoothing(
                       trend="add", seasonal="add" if sp > 1 else None, sp=sp if sp > 1 else None),
                   family="Statistical"),
    ForecasterSpec("ets", "ETS (automatic)", _sk("sktime.forecasting.ets.AutoETS", auto=True, sp=_SEAS, n_jobs=1),
                   family="Statistical", speed="medium"),
    ForecasterSpec("theta", "Theta", _sk("sktime.forecasting.theta.ThetaForecaster", sp=_SEAS), family="Statistical"),
    ForecasterSpec("stlf", "STL forecaster", _sk("sktime.forecasting.trend.STLForecaster", sp=lambda sp: max(sp, 2)),
                   family="Statistical", notes="Needs a seasonal period ≥ 2."),
    ForecasterSpec("croston", "Croston", _sk("sktime.forecasting.croston.Croston"), family="Statistical",
                   notes="For intermittent demand (many zeros)."),
    ForecasterSpec("bats", "BATS", _sk("sktime.forecasting.bats.BATS", sp=_SEAS, n_jobs=1, use_box_cox=False),
                   family="Statistical", speed="slow"),
    ForecasterSpec("tbats", "TBATS", _sk("sktime.forecasting.tbats.TBATS", sp=lambda sp: [sp] if sp > 1 else None,
                                         n_jobs=1, use_box_cox=False), family="Statistical", speed="slow"),
]}
for _k, _n, _fam, _spd in [("lr", "Linear", "Linear", "fast"), ("en", "Elastic Net", "Linear", "fast"),
                           ("ridge", "Ridge", "Linear", "fast"), ("lasso", "Lasso", "Linear", "fast"),
                           ("llar", "Lasso Least Angle", "Linear", "fast"), ("br", "Bayesian Ridge", "Linear", "fast"),
                           ("huber", "Huber", "Linear", "fast"), ("omp", "Orthogonal Matching Pursuit", "Linear", "fast"),
                           ("par", "Passive Aggressive", "Linear", "fast"), ("knn", "K-Nearest Neighbours", "Neighbours", "fast"),
                           ("dt", "Decision Tree", "Tree", "fast"), ("rf", "Random Forest", "Tree", "medium"),
                           ("et", "Extra Trees", "Tree", "medium"), ("gbr", "Gradient Boosting", "Boosting", "medium"),
                           ("ada", "AdaBoost", "Boosting", "medium"), ("lightgbm", "LightGBM", "Boosting", "fast"),
                           ("xgboost", "XGBoost", "Boosting", "fast"), ("catboost", "CatBoost", "Boosting", "medium")]:
    REGISTRY[f"{_k}_cds_dt"] = ForecasterSpec(
        f"{_k}_cds_dt", f"{_n} (reduction)", _reduction(_reg(_k)), family=f"ML reduction: {_fam}",
        speed=_spd, notes="Conditional deseasonalise + detrend, then a regressor on lagged values (recursive).",
        params={"window_length": (None, "int", 2, 60)})
try:
    import prophet  # noqa: F401
    REGISTRY["prophet"] = ForecasterSpec("prophet", "Prophet", _sk("sktime.forecasting.fbprophet.Prophet"),
                                         family="Statistical", speed="medium")
except ImportError:
    pass

DEFAULT = ["naive", "snaive", "polytrend", "arima", "exp_smooth", "ets", "theta", "stlf",
           "ridge_cds_dt", "lightgbm_cds_dt"]


def get(key: str) -> ForecasterSpec:
    if key not in REGISTRY:
        raise KeyError(f"Unknown forecaster '{key}'. Available: {list(REGISTRY)}")
    return REGISTRY[key]


# --------------------------------------------------------------------------- data
@dataclass
class SeriesData:
    y: pd.Series                   # PeriodIndex when a regular frequency exists, else RangeIndex
    freq: Optional[str]
    sp_suggested: int
    issues: list
    n_imputed: int


def prepare_series(df: pd.DataFrame, target: str, date_col: Optional[str] = None,
                   impute: str = "interpolate") -> SeriesData:
    issues = []
    d = df[[c for c in (date_col, target) if c]].copy()
    if not pd.api.types.is_numeric_dtype(d[target]):
        raise ValueError(f"Target '{target}' must be numeric.")
    freq = None
    if date_col:
        d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
        bad = d[date_col].isna().sum()
        if bad:
            issues.append(f"{bad} unparseable date(s) removed.")
            d = d.dropna(subset=[date_col])
        dup = d[date_col].duplicated().sum()
        if dup:
            issues.append(f"{dup} duplicate timestamp(s): values averaged.")
            d = d.groupby(date_col, as_index=False)[target].mean()
        d = d.sort_values(date_col)
        freq = pd.infer_freq(d[date_col]) if len(d) >= 3 else None
        if freq is None and len(d) >= 3:
            # try to regularise: most common gap
            gaps = d[date_col].diff().dropna().value_counts()
            for f in ("MS", "M", "QS", "W", "D", "h"):
                rng = pd.date_range(d[date_col].iloc[0], d[date_col].iloc[-1], freq=f)
                if len(rng) >= len(d) and set(d[date_col]) <= set(rng):
                    missing = len(rng) - len(d)
                    issues.append(f"Irregular timestamps: {missing} missing period(s) inserted at frequency '{f}'.")
                    d = d.set_index(date_col).reindex(rng).rename_axis(date_col).reset_index()
                    freq = f
                    break
            if freq is None:
                issues.append(f"No regular frequency found (most common gap {gaps.index[0]}); using a sequence index.")
        if freq:
            base = freq.split("-")[0]
            pfreq = {"MS": "M", "QS": "Q", "AS": "Y", "YS": "Y", "A": "Y"}.get(base, base)
            try:
                idx = pd.PeriodIndex(d[date_col], freq=pfreq)
            except Exception:  # noqa: BLE001
                idx = pd.RangeIndex(len(d)); freq = None
        else:
            idx = pd.RangeIndex(len(d))
    else:
        idx = pd.RangeIndex(len(d))
    y = pd.Series(d[target].to_numpy(float), index=idx, name=target)
    n_missing = int(y.isna().sum())
    if n_missing:
        if impute == "interpolate":
            y = y.interpolate(limit_direction="both")
        elif impute == "ffill":
            y = y.ffill().bfill()
        issues.append(f"{n_missing} missing value(s) filled by {impute}.")
    sp = suggest_sp(y, freq)
    return SeriesData(y, freq, sp, issues, n_missing)


def suggest_sp(y: pd.Series, freq: Optional[str]) -> int:
    """Seasonal period from the frequency, confirmed by the autocorrelation peak when possible."""
    base = (freq or "").split("-")[0]
    sp = FREQ_SP.get(base, 1)
    if len(y) >= 3 * max(sp, 2) and sp > 1:
        return sp
    from statsmodels.tsa.stattools import acf
    if len(y) < 12:
        return 1
    a = acf(np.diff(y.to_numpy(float)), nlags=min(len(y) // 3, 60), fft=True)
    lag = int(np.argmax(a[2:]) + 2) if len(a) > 3 else 1
    return lag if a[lag] > 0.3 else 1


# --------------------------------------------------------------------------- metrics
def ts_metrics(y_train, y_test, y_pred, sp: int) -> dict:
    yt, yp, ytr = np.asarray(y_test, float), np.asarray(y_pred, float), np.asarray(y_train, float)
    m = max(sp, 1)
    scale_mae = np.mean(np.abs(ytr[m:] - ytr[:-m])) if len(ytr) > m else np.nan
    scale_mse = np.mean((ytr[m:] - ytr[:-m]) ** 2) if len(ytr) > m else np.nan
    err = yt - yp
    out = {"mase": np.mean(np.abs(err)) / scale_mae if scale_mae else np.nan,
           "rmsse": np.sqrt(np.mean(err ** 2) / scale_mse) if scale_mse else np.nan,
           "mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2))),
           "smape": float(np.mean(2 * np.abs(err) / np.maximum(np.abs(yt) + np.abs(yp), 1e-12)))}
    if np.all(np.abs(yt) > 1e-12):
        out["mape"] = float(np.mean(np.abs(err / yt)))
    return out


TS_METRIC_LABELS = {"mase": "MASE", "rmsse": "RMSSE", "mae": "MAE", "rmse": "RMSE", "mape": "MAPE",
                    "smape": "sMAPE", "coverage": "Interval coverage"}


# --------------------------------------------------------------------------- validation
@dataclass
class TSDesign:
    fh: int                         # forecast horizon
    sp: int                         # seasonal period
    n_folds: int = 3                # rolling origins on the development part
    holdout: bool = True
    coverage: float = 0.9
    random_state: int = 42

    def split(self, n: int) -> tuple[int, list]:
        """(development length, [(train_end, test_end), ...] rolling origins within development)."""
        n_dev = n - self.fh if self.holdout else n
        folds = []
        for i in range(self.n_folds, 0, -1):
            cut = n_dev - i * self.fh
            folds.append((cut, cut + self.fh))
        return n_dev, folds

    def validate(self, n: int) -> list[str]:
        problems = []
        n_dev, folds = self.split(n)
        min_train = max(2 * self.sp, 10)
        if folds and folds[0][0] < min_train:
            problems.append(f"Too little history: the first CV origin has {folds[0][0]} observations; need at least "
                            f"{min_train} (2 seasonal cycles). Reduce the horizon or the number of folds.")
        return problems


def supports_intervals(key: str, sp: int = 12) -> bool:
    try:
        return bool(make(key, sp).get_tag("capability:pred_int", False))
    except Exception:  # noqa: BLE001
        return False


def make(key: str, sp: int, random_state: int = 0, params: Optional[dict] = None):
    return get(key).factory(sp, random_state, dict(params or {}))


def _fit_predict(key, y_train, fh, sp, rs, params=None, coverage=None):
    f = make(key, sp, rs, params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f.fit(y_train)
        h = list(range(1, fh + 1))
        pred = f.predict(fh=h)
        pi = None
        if coverage and f.get_tag("capability:pred_int", False):
            try:
                pi = f.predict_interval(fh=h, coverage=coverage)
            except Exception:  # noqa: BLE001
                pi = None
    return f, pred, pi


def cross_validate_forecaster(y: pd.Series, key: str, design: TSDesign, params=None) -> dict:
    n_dev, folds = design.split(len(y))
    rows, preds = [], []
    for i, (a, b) in enumerate(folds):
        _, pred, _ = _fit_predict(key, y.iloc[:a], design.fh, design.sp, design.random_state, params)
        m = ts_metrics(y.iloc[:a], y.iloc[a:b], pred.values, design.sp)
        rows.append({"fold": i + 1, "train_end": str(y.index[a - 1]), "n_train": a, **m})
        preds.append(pd.DataFrame({"fold": i + 1, "time": y.index[a:b].astype(str), "actual": y.iloc[a:b].values,
                                   "forecast": pred.values}))
    return {"folds": pd.DataFrame(rows), "predictions": pd.concat(preds, ignore_index=True)}


def compare_forecasters(y: pd.Series, keys: list, design: TSDesign, progress=None,
                        time_budget_s: Optional[float] = None) -> tuple[pd.DataFrame, dict, dict]:
    rows, fail, cvs = [], {}, {}
    t0 = time.time()
    for i, k in enumerate(keys):
        if progress:
            progress(i, len(keys), get(k).name)
        if time_budget_s and time.time() - t0 > time_budget_s:
            fail[k] = "skipped: time budget exhausted"; continue
        t = time.time()
        try:
            cv = cross_validate_forecaster(y, k, design)
        except Exception as e:  # noqa: BLE001
            fail[k] = f"{type(e).__name__}: {str(e)[:200]}"; continue
        f = cv["folds"]
        row = {"key": k, "Model": get(k).name, "family": get(k).family}
        for m in ("mase", "rmsse", "mae", "rmse", "mape", "smape"):
            if m in f:
                row[f"{m}_mean"], row[f"{m}_sd"] = f[m].mean(), f[m].std(ddof=1)
        row["fit_time_s"] = time.time() - t
        rows.append(row)
        cvs[k] = cv
    lb = pd.DataFrame(rows)
    if len(lb):
        lb = lb.sort_values("mase_mean", na_position="last").reset_index(drop=True)
        lb.insert(0, "rank", np.arange(1, len(lb) + 1))
        if "snaive" in set(lb["key"]):
            base = lb.loc[lb["key"] == "snaive", "mase_mean"].iloc[0]
            lb["beats_seasonal_naive"] = lb["mase_mean"] < base
    return lb, fail, cvs


def evaluate_holdout(y: pd.Series, key: str, design: TSDesign, params=None) -> dict:
    if not design.holdout:
        raise ValueError("No holdout was reserved.")
    n_dev = len(y) - design.fh
    f, pred, pi = _fit_predict(key, y.iloc[:n_dev], design.fh, design.sp, design.random_state, params,
                               design.coverage)
    m = ts_metrics(y.iloc[:n_dev], y.iloc[n_dev:], pred.values, design.sp)
    out = pd.DataFrame({"time": y.index[n_dev:].astype(str), "actual": y.iloc[n_dev:].values,
                        "forecast": pred.values})
    if pi is not None:
        lo, hi = pi.iloc[:, 0].values, pi.iloc[:, 1].values
        out["lower"], out["upper"] = lo, hi
        m["coverage"] = float(np.mean((out["actual"] >= lo) & (out["actual"] <= hi)))
    return {"metrics": m, "table": out, "n_dev": n_dev}


def final_forecast(y: pd.Series, key: str, design: TSDesign, params=None, horizon: Optional[int] = None) -> dict:
    """Refit on the full series (development + holdout) and forecast beyond the last observation."""
    h = horizon or design.fh
    f, pred, pi = _fit_predict(key, y, h, design.sp, design.random_state, params, design.coverage)
    out = pd.DataFrame({"time": pred.index.astype(str), "forecast": pred.values})
    if pi is not None:
        out["lower"], out["upper"] = pi.iloc[:, 0].values, pi.iloc[:, 1].values
    return {"model": f, "table": out}


def diagnostics(y: pd.Series, sp: int) -> dict:
    """Stationarity tests, STL decomposition and ACF/PACF for the Analyze step."""
    from statsmodels.tsa.seasonal import STL
    from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf
    v = y.to_numpy(float)
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            out["adf_p"] = float(adfuller(v, autolag="AIC")[1])
        except Exception:  # noqa: BLE001
            out["adf_p"] = np.nan
        try:
            out["kpss_p"] = float(kpss(v, regression="c", nlags="auto")[1])
        except Exception:  # noqa: BLE001
            out["kpss_p"] = np.nan
        nl = int(min(len(v) // 2 - 1, max(24, 2 * sp)))
        out["acf"], out["pacf"] = acf(v, nlags=nl, fft=True), pacf(v, nlags=min(nl, len(v) // 2 - 1))
        if sp > 1 and len(v) >= 2 * sp + 1:
            r = STL(v, period=sp, robust=True).fit()
            out["stl"] = pd.DataFrame({"trend": r.trend, "seasonal": r.seasonal, "resid": r.resid},
                                      index=y.index.astype(str))
            var_r = np.var(r.resid)
            out["seasonal_strength"] = float(max(0, 1 - var_r / np.var(r.seasonal + r.resid)))
            out["trend_strength"] = float(max(0, 1 - var_r / np.var(r.trend + r.resid)))
    return out


def ljung_box(residuals, lags: int = 10) -> float:
    from statsmodels.stats.diagnostic import acorr_ljungbox
    r = pd.Series(residuals).dropna()
    if len(r) <= lags + 1:
        return np.nan
    return float(acorr_ljungbox(r, lags=[lags], return_df=True)["lb_pvalue"].iloc[0])
