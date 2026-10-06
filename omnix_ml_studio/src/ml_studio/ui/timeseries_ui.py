"""Time series: Data, Setup, Compare, Create & tune, Analyze, Validate, Forecast & deploy."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from ..core import model_manager as MM
from ..core import timeseries as TS
from ..visualization import unsup_plots as UP
from . import jobs
from . import shell as SH
from .components import downloads, model_picker, fig_out
from .unsup_common import NONE, clear, source_card, spec_param_editor

FAMILIES = ["Baseline", "Statistical"] + sorted({s.family for s in TS.REGISTRY.values()
                                                 if s.family.startswith("ML reduction")})
TL = TS.TS_METRIC_LABELS
METRICS = ["mase", "rmsse", "mae", "rmse", "mape", "smape"]


def _y(S):
    return S.u["series"].y


# =========================================================================== DATA
def data_step(S, module, main, side):
    P = SH.panel(side, "data")
    if not source_card(S, module, P):
        SH.empty(main, "No series yet", "Upload a table with one row per time point: a date column and a numeric "
                 "value column. Rows are sorted by date; missing periods are inserted and filled.")
        return
    df = S.df
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    dates = [c for c in df.columns if c not in num or "date" in c.lower() or "time" in c.lower()]
    c = SH.card(P, f"{module.key}_roles", "Series")
    if not num:
        c.error("No numeric column to forecast."); return
    tgt = c.selectbox("Value to forecast", num, key=f"{module.key}_target")
    dopt = [NONE] + [x for x in df.columns if x != tgt]
    dg = next((x for x in dates if x != tgt), NONE)
    dcol = c.selectbox("Date / time column", dopt, index=dopt.index(dg) if dg in dopt else 0, key=f"{module.key}_date",
                       help="Without one, rows are taken as equally spaced in file order.")
    imp = c.selectbox("Fill missing values", ["interpolate", "ffill"], key=f"{module.key}_impute",
                      format_func=lambda x: {"interpolate": "linear interpolation", "ffill": "carry last value forward"}[x])
    if SH.run_button(P, module, "Confirm series", key_suffix="_roles"):
        clear(S, "data")
        try:
            S.u["series"] = TS.prepare_series(df, tgt, None if dcol == NONE else dcol, imp)
        except ValueError as e:
            main.error(str(e)); return
        S.log("Series", f"{tgt}, {len(S.u['series'].y)} points")
        SH.advance(module, "setup")
        st.rerun()
    sd = S.u.get("series")
    if sd is None:
        with main.expander("Preview", expanded=True):
            st.dataframe(df.head(30), width="stretch", height=260)
        main.info("Choose the value and date columns in the panel and confirm them.")
        return
    y = sd.y
    SH.kpis(main, [("observations", f"{len(y):,}"), ("frequency", sd.freq or "none (sequence)"),
                   ("suggested seasonal period", str(sd.sp_suggested)), ("filled values", str(sd.n_imputed))])
    for i in sd.issues:
        main.warning(i)
    fig_out(main, UP.series_plot(y, f"{y.name}"))


# =========================================================================== SETUP
def setup_step(S, module, main, side):
    sd = S.u["series"]
    y = sd.y
    n = len(y)
    P = SH.panel(side, "setup")
    d0 = S.u.get("design")
    c = SH.card(P, f"{module.key}_fh", "Forecast", "The horizon is how many periods ahead you need to forecast.")
    fh = int(c.number_input("Horizon", 1, max(1, n // 3), int(d0.fh if d0 else min(max(sd.sp_suggested, 6), n // 5)),
                            key=f"{module.key}_fhn"))
    sp = int(c.number_input("Seasonal period", 1, max(1, n // 2), int(d0.sp if d0 else sd.sp_suggested),
                            key=f"{module.key}_sp", help="12 for monthly data with yearly seasonality, 7 for daily "
                                                         "data with weekly seasonality, 1 for none."))
    c = SH.card(P, f"{module.key}_val", "Validation design", "Time order is preserved: models only see the past.")
    folds = int(c.number_input("Rolling origins (CV folds)", 1, 10, int(d0.n_folds if d0 else 3), key=f"{module.key}_folds"))
    hold = c.checkbox("Seal the last horizon as a holdout", value=d0.holdout if d0 else True, key=f"{module.key}_hold")
    cov = float(c.slider("Prediction-interval coverage", 0.5, 0.99, float(d0.coverage if d0 else 0.9), 0.01,
                         key=f"{module.key}_cov"))
    design = TS.TSDesign(fh, sp, folds, hold, cov)
    problems = design.validate(n)
    for p in problems:
        P.error(p)
    if SH.run_button(P, module, "Confirm setup", disabled=bool(problems), key_suffix="_setup"):
        clear(S, "setup")
        S.u["design"] = design
        S.u["diag"] = TS.diagnostics(y.iloc[: design.split(n)[0]], sp)
        S.log("Setup", f"horizon {fh}, sp {sp}, {folds} origins, holdout {'yes' if hold else 'no'}")
        SH.advance(module, "compare")
        st.rerun()
    n_dev, fl = design.split(n)
    fig_out(main, UP.series_plot(y, f"{y.name}: development and holdout", n_dev if hold else None))
    if fl and fl[0][0] > 0:
        fig_out(main, UP.cv_origins(n, fl, n_dev, fh, y.index))
    if S.u.get("design") is None:
        main.info("Confirm the setup to seal the holdout and compute series diagnostics.")
        return
    dg = S.u["diag"]
    main.success(f"Setup confirmed. {n_dev} development points" + (f", last {fh} sealed." if hold else "."))
    SH.section(main, "Series diagnostics (development part)")
    items = [("ADF p (unit root)", f"{dg['adf_p']:.3f}"), ("KPSS p (stationary)", f"{dg['kpss_p']:.3f}")]
    if "seasonal_strength" in dg:
        items += [("seasonal strength", f"{dg['seasonal_strength']:.2f}"), ("trend strength", f"{dg['trend_strength']:.2f}")]
    SH.kpis(main, items)
    main.caption("ADF p < 0.05 suggests no unit root; KPSS p < 0.05 suggests non-stationarity. Strengths near 1 mean "
                 "the component dominates the remainder (STL).")


# =========================================================================== COMPARE
def compare_step(S, module, main, side):
    y, design = _y(S), S.u["design"]
    P = SH.panel(side, "compare")
    specs = dict(TS.REGISTRY)
    c = SH.card(P, f"{module.key}_algos", "Forecasters", f"{len(specs)} available (sktime).")
    keys = model_picker(c, f"{module.key}_pick", specs, FAMILIES, TS.DEFAULT, columns=1)
    c2 = SH.card(P, f"{module.key}_budget", "Budget")
    budget = c2.number_input("Time budget (seconds, 0 = none)", 0, 3600, 0, 30, key=f"{module.key}_budgetn")
    if "snaive" not in keys:
        c.caption("Seasonal naive is added as the reference for MASE.")
    SH.kpis(main, [("development points", str(design.split(len(y))[0])), ("horizon", str(design.fh)),
                   ("seasonal period", str(design.sp)), ("origins", str(design.n_folds))])
    jkey = f"{module.key}_compare"

    def apply(job):
        lb, fail, cvs = job.result
        clear(S, "compare")
        S.u.update(lb=lb, fail=fail, cvs=cvs)
        S.log("Compare", f"{len(lb)} forecasters")
    job = jobs.collect(jkey, apply, (id(y), repr(design)))
    busy = job is not None and job.running
    if SH.run_button(P, module, "Running in the background…" if busy else f"Compare {len(keys)} forecasters",
                     disabled=not keys or busy, key_suffix="_compare"):
        ks = list(dict.fromkeys((["snaive"] if "snaive" not in keys else []) + keys))

        def work(job, ks=tuple(ks), budget=budget):
            return jobs.run_stoppable(job, TS.compare_forecasters, y, list(ks), design, time_budget_s=budget or None,
                                      progress=jobs.stoppable_progress(job))
        jobs.start(jkey, f"Comparing {len(ks)} forecasters", work, signature=(id(y), repr(design)))
        st.rerun()
    if job is not None and job.status == "error":
        main.error(f"The comparison did not finish: {job.error}")
    with main:
        jobs.monitor(jkey)
    if busy:
        return
    lb = S.u.get("lb")
    if lb is None:
        SH.empty(main, "No comparison yet", "Each forecaster is refitted at every rolling origin on the past only and "
                 "scored on the next horizon. MASE below 1 means better than the seasonal-naive forecast.")
        return
    if lb.empty:
        main.error("Every forecaster failed: " + "; ".join(list(S.u["fail"].values())[:3])); return
    SH.section(main, "Leaderboard", "Mean across rolling origins of the development part; lower is better for every "
                                    "metric. MASE and RMSSE are scaled by the in-sample seasonal-naive error.")
    tab = pd.DataFrame({"Model": lb["Model"].values})
    for m in METRICS:
        if f"{m}_mean" in lb:
            tab[TL[m]] = lb[f"{m}_mean"].values
    tab["Time (s)"] = lb["fit_time_s"].values
    if "beats_seasonal_naive" in lb:
        tab["Beats seasonal naive"] = lb["beats_seasonal_naive"].map({True: "yes", False: "no"}).values
    fmt = {TL[m]: "{:.3f}" for m in METRICS if TL[m] in tab}
    fmt.update({"MAPE": "{:.1%}", "sMAPE": "{:.1%}", "Time (s)": "{:.2f}"})
    sty = tab.style.format(fmt, na_rep="—")
    for m in METRICS:
        if TL[m] in tab:
            sty = sty.highlight_min(subset=[TL[m]], props=f"background-color:{module.tint}; font-weight:600; color:{module.color}")
    main.dataframe(sty, hide_index=True, width="stretch")
    if S.u["fail"]:
        with main.expander(f"{len(S.u['fail'])} forecaster(s) failed or were skipped"):
            for k, msg in S.u["fail"].items():
                st.markdown(f"**{TS.get(k).name}**: {msg}")
    pick = main.selectbox("Show CV forecasts for", list(lb["key"]), format_func=lambda k: TS.get(k).name,
                          key=f"{module.key}_cvshow")
    fig_out(main, UP.cv_forecasts(y.iloc[: design.split(len(y))[0]], S.u["cvs"][pick]["predictions"]))
    downloads(main, lb, "forecaster_comparison", f"{module.key}_lbdl")
    best = lb.iloc[0]
    if main.button(f"Create and tune {best['Model']}", key=f"{module.key}_tobest", type="primary"):
        st.session_state[f"{module.key}_cm_key"] = best["key"]
        SH.go_to(module, "create")
        st.rerun()


# =========================================================================== CREATE
def create_step(S, module, main, side):
    y, design = _y(S), S.u["design"]
    models = S.u.setdefault("models", {})
    P = SH.panel(side, "create")
    keys = list(TS.REGISTRY)
    if S.u.get("lb") is not None and len(S.u["lb"]):
        ranked = list(S.u["lb"]["key"])
        keys = ranked + [k for k in keys if k not in ranked]
    ck = f"{module.key}_cm_key"
    if st.session_state.get(ck) not in keys:
        st.session_state[ck] = keys[0]
    c = SH.card(P, f"{module.key}_cm", "Forecaster")
    key = c.selectbox("Forecaster", keys, format_func=lambda k: TS.get(k).name, key=ck, label_visibility="collapsed")
    spec = TS.get(key)
    if spec.notes:
        c.caption(spec.notes)
    c = SH.card(P, f"{module.key}_hp", "Parameters")
    params = spec_param_editor(c, f"{module.key}_pe_{key}", spec.params) if spec.params else {}
    if not spec.params:
        c.caption("No tunable parameters exposed; automatic methods (Auto ARIMA, ETS, BATS) select their own.")
    if SH.run_button(P, module, "Create model", key_suffix="_create"):
        try:
            with main.spinner(f"Cross-validating {spec.name}…"):
                cv = TS.cross_validate_forecaster(y, key, design, params)
        except Exception as e:  # noqa: BLE001
            main.error(f"{spec.name} failed: {type(e).__name__}: {e}"); return
        desc = ", ".join(f"{k}={v}" for k, v in params.items())
        name = spec.name + (f" ({desc})" if desc else "")
        models[name] = {"key": key, "params": params, "cv": cv}
        st.session_state[f"{module.key}_cm_view"] = name
        S.log("Create", f"{name}: MASE {cv['folds']['mase'].mean():.3f}")
        st.rerun()
    if not models:
        SH.empty(main, "No model created yet", "Choose a forecaster in the panel and press Create model. It is "
                 "cross-validated on the same rolling origins as the comparison.")
        return
    SH.section(main, "Created models")
    rows = [{"Model": n, **{TL[m]: r["cv"]["folds"][m].mean() for m in METRICS if m in r["cv"]["folds"]},
             "final": "✓" if S.u.get("chosen") == n else ""} for n, r in models.items()]
    main.dataframe(pd.DataFrame(rows).round(4), hide_index=True, width="stretch")
    names = list(models)
    vk = f"{module.key}_cm_view"
    if st.session_state.get(vk) not in names:
        st.session_state[vk] = names[-1]
    a, b = main.columns([3, 1], vertical_alignment="bottom")
    view = a.selectbox("Model", names, key=vk)
    is_final = S.u.get("chosen") == view
    if b.button("Final model ✓" if is_final else "Set as final model", key=f"{module.key}_setfinal",
                type="secondary" if is_final else "primary", use_container_width=True, disabled=is_final):
        for k in ("holdout", "final", "bundle", "meta", "deploy"):
            S.u.pop(k, None)
        S.u["chosen"] = view
        S.log("Final model", view)
        SH.advance(module, "analyze")
        st.rerun()
    r = models[view]
    main.dataframe(r["cv"]["folds"].round(4), hide_index=True, width="stretch")
    fig_out(main, UP.cv_forecasts(y.iloc[: design.split(len(y))[0]], r["cv"]["predictions"]))


# =========================================================================== ANALYZE
def analyze_step(S, module, main, side):
    y, design, dg = _y(S), S.u["design"], S.u["diag"]
    m = S.u["models"][S.u["chosen"]]
    n_dev = design.split(len(y))[0]
    P = SH.panel(side, "analyze")
    c = SH.card(P, f"{module.key}_plot", "Plot")
    all_ = SH.picker(c, f"{module.key}_an_all", "Show", ["One plot", "All plots"]) == "All plots"
    options = ["Decomposition (STL)", "ACF and PACF", "CV forecasts", "CV residuals"]
    selected = options if all_ else [c.radio("Plot", options, key=f"{module.key}_plotsel", label_visibility="collapsed")]
    main.markdown(f"**{S.u['chosen']}** · development part ({n_dev} points)")
    for plot in selected:
        if all_:
            SH.section(main, plot)
        if plot.startswith("Decomposition"):
            if "stl" in dg:
                fig_out(main, UP.stl_plot(dg["stl"], y.iloc[:n_dev]))
            else:
                main.info("STL needs a seasonal period of at least 2 and two full cycles of data.")
        elif plot == "ACF and PACF":
            fig_out(main, UP.acf_plot(dg["acf"], dg["pacf"], n_dev))
            main.caption("Bars outside the shaded band (±1.96/√n) indicate autocorrelation at that lag.")
        elif plot == "CV forecasts":
            fig_out(main, UP.cv_forecasts(y.iloc[:n_dev], m["cv"]["predictions"]))
        else:
            p = m["cv"]["predictions"]
            r = (p["actual"] - p["forecast"]).to_numpy()
            fig_out(main, UP.residual_panel(r, "Out-of-sample residuals across all origins"))
            lb_p = TS.ljung_box(r, min(10, max(1, len(r) // 3)))
            main.markdown(f"Mean residual **{r.mean():.3g}** (bias) · Ljung–Box p = **{lb_p:.3f}**")
            main.caption("Residuals are pooled across origins, so autocorrelation tests are approximate. A clearly non-zero "
                         "mean indicates systematic over- or under-forecasting.")


# =========================================================================== VALIDATE
def validate_step(S, module, main, side):
    y, design = _y(S), S.u["design"]
    m = S.u["models"][S.u["chosen"]]
    P = SH.panel(side, "validate")
    c = SH.card(P, f"{module.key}_ho", "Sealed holdout", "Fits on the development part and forecasts the sealed last "
                "horizon once.")
    if not design.holdout:
        c.info("No holdout was sealed in Setup. The cross-validation estimate is the only one available.")
    if SH.run_button(P, module, "Evaluate on holdout", disabled=not design.holdout, key_suffix="_holdout"):
        with main.spinner("Forecasting the holdout…"):
            try:
                S.u["holdout"] = (S.u["chosen"], TS.evaluate_holdout(y, m["key"], design, m["params"]))
                S.u["holdout_n"] = S.u.get("holdout_n", 0) + 1
            except Exception as e:  # noqa: BLE001
                main.error(f"Holdout evaluation failed: {e}"); return
        S.log("Holdout", S.u["chosen"])
        st.rerun()
    if S.u.get("holdout_n", 0) > 1:
        P.warning(f"The holdout has been evaluated {S.u['holdout_n']} times. Picking the model that scores best on it "
                  "makes the estimate optimistic.")
    h = S.u.get("holdout")
    if not h or h[0] != S.u["chosen"]:
        SH.empty(main, "Holdout still sealed", f"Final model: {S.u['chosen']}. Evaluate once to get an estimate that "
                 "played no part in choosing the model.")
        return
    r = h[1]
    met = r["metrics"]
    SH.kpis(main, [(f"holdout {TL[k]}", f"{met[k]:.1%}" if k in ("mape", "smape", "coverage") else f"{met[k]:.3f}")
                   for k in ("mase", "rmse", "mape", "coverage") if k in met])
    if "coverage" in met:
        main.caption(f"Nominal interval coverage {design.coverage:.0%}; observed {met['coverage']:.0%} of {design.fh} "
                     "points. With a short horizon, coverage is very imprecise.")
    else:
        main.caption("This forecaster provides no prediction intervals.")
    fig_out(main, UP.forecast_plot(y.iloc[: r["n_dev"]], r["table"], "Holdout forecast versus actual",
                                       history=min(r["n_dev"], 6 * max(design.fh, design.sp))))
    main.dataframe(r["table"].round(4), hide_index=True, width="stretch")
    downloads(main, r["table"], "holdout_forecast", f"{module.key}_hodl")


# =========================================================================== FORECAST & DEPLOY
def deploy_step(S, module, main, side):
    y, design = _y(S), S.u["design"]
    name = S.u["chosen"]
    m = S.u["models"][name]
    P = SH.panel(side, "deploy")
    c = SH.card(P, f"{module.key}_fin", "Final forecast", "Refits on the whole series, holdout included, and forecasts "
                "beyond the last observation.")
    h = int(c.number_input("Periods ahead", 1, max(1, len(y) // 2), design.fh, key=f"{module.key}_finh"))
    if SH.run_button(P, module, "Fit on all data and forecast", key_suffix="_finalize"):
        with main.spinner("Fitting and forecasting…"):
            try:
                r = TS.final_forecast(y, m["key"], design, m["params"], h)
            except Exception as e:  # noqa: BLE001
                main.error(f"Forecast failed: {e}"); return
        ho = S.u.get("holdout")
        meta = {"module": "timeseries", "task": "forecast", "model": name, "model_key": m["key"],
                "params": {k: str(v) for k, v in m["params"].items()}, "series": y.name, "n_train": len(y),
                "last_period": str(y.index[-1]), "freq": S.u["series"].freq, "sp": design.sp, "coverage": design.coverage,
                "fingerprint": MM.dataset_fingerprint(y.to_frame()),
                "holdout_metrics": {k: float(v) for k, v in ho[1]["metrics"].items()} if ho and ho[0] == name else None}
        S.u["final"] = r
        S.u["bundle"] = MM.make_bundle(r["model"], meta)
        S.u["meta"], S.u["deploy"] = MM.read_metadata(S.u["bundle"]), r["model"]
        S.log("Forecast", f"{name}, {h} periods")
        st.rerun()
    c = SH.card(P, f"{module.key}_load", "Or load a bundle")
    up = c.file_uploader("Bundle", type=["omnix", "zip"], key=f"{module.key}_bundleup", label_visibility="collapsed")
    if up is not None and c.button("Load bundle", key=f"{module.key}_loadb", use_container_width=True):
        try:
            mdl, _, meta = MM.load_bundle(up.getvalue())
            if meta.get("module") != "timeseries":
                raise MM.BundleError("This bundle is not a time-series model.")
            S.u["deploy"], S.u["meta"], S.u["bundle"] = mdl, meta, up.getvalue()
            pred = mdl.predict(fh=list(range(1, h + 1)))
            S.u["final"] = {"model": mdl, "table": pd.DataFrame({"time": pred.index.astype(str), "forecast": pred.values})}
            st.rerun()
        except MM.BundleError as e:
            c.error(str(e))
    r = S.u.get("final")
    if r is None:
        SH.empty(main, "No forecast yet", "Fit the final model on the whole series to forecast the future and "
                 "download a signed model bundle.")
        return
    meta = S.u["meta"]
    ho = meta.get("holdout_metrics")
    SH.kpis(main, [("model", meta["model"]), ("fitted through", meta["last_period"]),
                   ("holdout MASE", f"{ho['mase']:.3f}" if ho else "not evaluated")])
    fig_out(main, UP.forecast_plot(y, r["table"], f"Forecast: next {len(r['table'])} periods",
                                       history=min(len(y), 6 * max(design.fh, design.sp))))
    main.caption("Intervals (where shown) assume the model's error structure holds; they do not include uncertainty "
                 "from model selection or future structural change.")
    main.dataframe(r["table"].round(4), hide_index=True, width="stretch")
    downloads(main, r["table"], "forecast", f"{module.key}_fcdl")
    if S.u.get("bundle"):
        main.download_button("Download model bundle", S.u["bundle"], f"forecast_{meta['fingerprint'][:8]}.omnix",
                             "application/zip", key=f"{module.key}_bundledl", type="primary")
    if not MM.key_is_persistent():
        main.caption("Bundles reload only while this server keeps running unless OMNIX_MODEL_KEY is set on the server.")
