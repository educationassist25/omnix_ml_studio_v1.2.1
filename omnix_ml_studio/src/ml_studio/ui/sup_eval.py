"""Supervised workflow — step 6 (Validate), step 7 (Explain), step 8 (Predict & deploy)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from ..core import model_manager as MM
from ..core.data_validation import DataLoadError, load_table
from ..core.explainability import (CAVEAT, aggregate_to_original, explain_shap, model_features_original, model_input,
                                   model_specific_importance, pdp_ice, permutation_importances, positive_output_index,
                                   shap_importance)
from ..core.feature_selection import to_original
from ..core.metrics import METRIC_LABELS, primary_metric
from ..core.model_comparison import registry
from ..core.validation import evaluate_holdout, external_validate
from ..visualization import classification_plots as CP
from ..visualization import regression_plots as RP
from ..visualization import validation_plots as VP
from ..visualization import xai_plots as XP
from . import shell as SH
from ..core import tasks as TK
from . import jobs
from .components import ci_table, downloads, feature_filter, fig_out, model_picker, protected, safe_slider
from .sup_model import fit_seconds, fmt_duration, grid_size

ERRORS = ["log_loss", "rmse", "mae", "mse", "medae", "mape"]


def perf_plots(main, task, y, y_pred, proba, classes, prefix, key):
    if task == "regression":
        c1, c2 = main.columns(2)
        fig_out(c1, RP.observed_vs_predicted(y, y_pred, f"{prefix}: observed vs predicted"),
                        key=f"{key}_ovp")
        fig_out(c2, RP.residuals(y, y_pred, f"{prefix}: residuals"), key=f"{key}_res")
        return
    c1, c2 = main.columns(2)
    if proba is not None:
        fig_out(c1, CP.roc(y, proba, classes, f"{prefix}: ROC"), key=f"{key}_roc")
        fig_out(c2, CP.pr(y, proba, classes, f"{prefix}: precision–recall"), key=f"{key}_pr")
    c3, c4 = main.columns(2)
    fig_out(c3, CP.confusion(y, y_pred, classes, f"{prefix}: confusion matrix"), key=f"{key}_cm")
    if proba is not None and task == "binary":
        fig_out(c4, CP.calibration(y, proba, title=f"{prefix}: calibration"), key=f"{key}_cal")


def _ensure_dev_fit(S, main):
    """Final pipeline fitted on development rows (the model the holdout evaluates)."""
    if S.final_model is None:
        S.final_model = protected(S, main, TK.fit, "Fitting the final model on the development set…",
                                  key=S.chosen["key"], params=S.chosen["params"])
    return S.final_model


# =========================================================================== VALIDATE
def validate_step(S, module, main, side):
    d, D, sp = S.design, S.data, S.split
    reg = registry(d.task)
    P = SH.panel(side, "validate")
    c = SH.card(P, f"{module.key}_vmode", "Validation")
    modes = ["Sealed holdout", "Nested CV", "External cohort"]
    mode = SH.picker(c, f"{module.key}_vmode", "Method", modes)
    pm = primary_metric(d.task)
    if mode == "Sealed holdout":
        c.caption("Fits the chosen model and preprocessing on the development set, then scores the sealed holdout "
                  "once, with bootstrap 95% CIs.")
        has = len(sp.test_idx) > 0
        if not has:
            c.info("No holdout was reserved in Setup. Use nested CV for an unbiased estimate.")
        if S.ledger and S.ledger.warning():
            c.warning(S.ledger.warning())
        if SH.run_button(P, module, "Evaluate on holdout", disabled=not has, key_suffix="_holdout"):
            with main.spinner("Fitting and bootstrapping…"):
                S.final_model = None
                pipe = _ensure_dev_fit(S, main)
                r = evaluate_holdout(pipe, D, sp.test_idx, d.task, S.val_cfg.n_bootstrap, S.val_cfg.random_state,
                                     S.ledger, S.chosen["name"])
            r["model_name"] = S.chosen["name"]
            S.holdout = r
            m = r["metrics"].set_index("metric")
            S.log("Holdout", f"{S.chosen['name']}: {METRIC_LABELS[pm]} = {m.loc[pm, 'estimate']:.3f}"
                  if pm in m.index else S.chosen["name"])
            st.rerun()
        _holdout_results(S, main)
    elif mode == "Nested CV":
        c.caption("Outer folds estimate how well the whole procedure (choose a model, tune it, select features) "
                  "generalises. Inner folds make those choices without seeing the outer test fold.")
        rows = c.radio("Rows", ["Development set (holdout stays sealed)", "All samples"], key=f"{module.key}_nrows")
        idx = sp.train_idx if rows.startswith("Development") else np.arange(len(D.y))
        specs = {k: s for k, s in reg.REGISTRY.items() if k != "dummy"}
        default = [S.chosen["key"]] + [k for k in reg.DEFAULT if k not in ("dummy", S.chosen["key"])][:2]
        with c.expander("Candidate models", expanded=False):
            keys = model_picker(st, f"{module.key}_npick", specs, reg.FAMILIES, default, columns=1)
        c.caption(f"{len(keys)} candidates: " + ", ".join(specs[k].name for k in keys[:4]) + (" …" if len(keys) > 4 else ""))
        it = int(c.slider("Tuning iterations per model per outer fold", 10, 300, 100, 10, key=f"{module.key}_nit",
                          help="100 or more is recommended. Cost grows linearly with iterations."))
        n_out, n_in = S.val_cfg.n_folds, S.val_cfg.inner_folds

        def n_set(k):
            g = grid_size(specs[k].space)
            return 1 if not specs[k].space else (min(it, g) if g else it)
        fits = n_out * sum(n_set(k) * n_in + 1 for k in keys)
        per = [fit_seconds(S, k) for k in keys]
        est = None if any(p is None for p in per) else \
            1.3 * n_out * sum(p * (n_set(k) * n_in + 1) for k, p in zip(keys, per))
        c.caption(f"About {fits:,} model fits ({n_out} outer × {n_in} inner folds); {fmt_duration(est)}.")
        if est and est > 1800:
            c.warning("This will take a long time. Fewer candidate models, a univariate pre-filter in Setup, or "
                      "a larger server will help more than fewer iterations.")
        if len(d.features) > 500 and not S.prep_cfg.univariate_k:
            c.warning("Many features and no univariate pre-filter: nested CV may take many minutes.")
        nkey = f"{module.key}_nested"
        sig = (id(S.data), repr(S.prep_cfg), repr(S.val_cfg), S.split.description)

        def apply(job):
            S.nested = job.result
            s = S.nested.summary.set_index("metric")
            sm = S.nested.settings["selection_metric"]
            S.log("Nested CV", f"{METRIC_LABELS[sm]} {s.loc[sm, 'mean']:.3f} ± {s.loc[sm, 'sd']:.3f} "
                               f"({job.elapsed / 60:.1f} min)")
        njob = jobs.collect(nkey, apply, sig)
        nbusy = njob is not None and njob.running
        if SH.run_button(P, module, "Running in the background…" if nbusy else "Run nested CV",
                         disabled=not keys or nbusy, key_suffix="_nested"):
            payload = {"data": S.data, "idx": idx, "task": d.task, "prep": S.prep_cfg, "val": S.val_cfg}

            def work(job, keys=tuple(keys), it=it):
                from ..core.runner import run_isolated
                job.progress(0, 1, f"{fits:,} model fits")
                r = run_isolated(payload, TK.nested, [("nested", {"keys": list(keys), "n_iter": it})], workers=1,
                                 should_stop=job.stopped)["nested"]
                if job.stopped():
                    return None
                if r.status != "ok":
                    raise RuntimeError(r.message.split("\n")[0])
                job.progress(1, 1, "done")
                return r.value
            jobs.start(nkey, "Nested cross-validation", work, signature=sig)
            st.rerun()
        if njob is not None and njob.status == "error":
            main.error(f"Nested CV did not finish: {njob.error}")
        with main:
            jobs.monitor(nkey)
        if nbusy:
            return
        _nested_results(S, main, reg)
    else:
        c.caption("Scores the development-fitted model, exactly as it is, on an independent cohort. Nothing is "
                  "refitted or recalibrated.")
        up = c.file_uploader("External dataset (same feature names)", type=["csv", "tsv", "txt", "xlsx"],
                             key=f"{module.key}_extup")
        allow = c.checkbox("Allow missing model features (imputed with development values)", key=f"{module.key}_extallow")
        if SH.run_button(P, module, "Run external validation", disabled=up is None, key_suffix="_external"):
            try:
                ext = load_table(up.getvalue(), up.name)
                pipe = _ensure_dev_fit(S, main)
                r = external_validate(pipe, D, ext, d, d.task, S.val_cfg.n_bootstrap, S.val_cfg.random_state,
                                      allow_missing_features=allow)
            except (DataLoadError, ValueError) as e:
                main.error(str(e)); return
            S.external = {"result": r, "file": up.name}
            S.log("External validation", up.name)
            st.rerun()
        _external_results(S, main)


def _holdout_results(S, main):
    d, D = S.design, S.data
    r = S.holdout
    if r is None:
        SH.empty(main, "Holdout still sealed", f"Final model: {S.chosen['name']}. Evaluate it once; every further "
                 "evaluation is recorded and makes the estimate optimistic if you use it to choose models.")
        return
    ci = r["metrics"]
    pm = primary_metric(d.task)
    m = ci.set_index("metric")
    top = [pm] + [x for x in (["balanced_accuracy", "mcc"] if d.task != "regression" else ["rmse", "mae"])
                  if x in m.index]
    SH.kpis(main, [(f"holdout {METRIC_LABELS.get(k, k)}",
                    f"{m.loc[k, 'estimate']:.3f}" + (f" [{m.loc[k, 'ci_low']:.2f}, {m.loc[k, 'ci_high']:.2f}]"
                                                     if pd.notna(m.loc[k, "ci_low"]) else "")) for k in top if k in m.index]
            + [("samples", str(r["n"]))])
    main.caption(f"{r['model_name']} · bootstrap unit: {ci.attrs.get('unit', 'sample')}, "
                 f"{S.val_cfg.n_bootstrap} resamples · evaluation {S.ledger.n_evaluations if S.ledger else 1} of this holdout")
    for w in r["warnings"]:
        main.warning(w)
    c1, c2 = main.columns([2, 3])
    c1.dataframe(ci_table(ci), hide_index=True, width="stretch")
    fig_out(c2, VP.ci_forest(ci.loc[~ci["metric"].isin(ERRORS)]))
    perf_plots(main, d.task, r["y_true"], r["y_pred"], r["proba"], D.classes, "Holdout", "ho")
    downloads(main, ci, "holdout_metrics", "hodl")
    if S.ledger and S.ledger.entries:
        with main.expander("Holdout ledger"):
            st.dataframe(pd.DataFrame(S.ledger.entries), hide_index=True, width="stretch")


def _nested_results(S, main, reg):
    r = S.nested
    if r is None:
        SH.empty(main, "No nested CV yet", "Nested CV is the most honest internal estimate when data are small or "
                 "no holdout was reserved. It reports the performance of the procedure, not of one fitted model.")
        return
    sm = r.settings["selection_metric"]
    s = r.summary.set_index("metric")
    SH.kpis(main, [(f"nested-CV {METRIC_LABELS.get(sm, sm)}", f"{s.loc[sm, 'mean']:.3f} ± {s.loc[sm, 'sd']:.3f}"),
                   ("outer folds", str(r.settings["n_outer"])), ("candidates", str(len(r.settings["models"])))])
    c1, c2 = main.columns([3, 2])
    fig_out(c1, VP.nested_outer(r.outer, sm))
    c2.markdown("**Inner-loop winner per outer fold**")
    c2.dataframe(r.selection_frequency.rename(index=lambda k: reg.REGISTRY[k].name).rename("folds won"),
                 width="stretch")
    if r.selection_frequency.size > 1:
        c2.caption("Different winners across folds means model choice is unstable at this sample size.")
    main.dataframe(r.summary.round(4), hide_index=True, width="stretch")
    show = r.outer.drop(columns=["inner_scores"]).copy()
    show["params"] = show["params"].astype(str)
    with main.expander("Outer-fold details"):
        st.dataframe(show.round(4), hide_index=True, width="stretch")
    downloads(main, show, "nested_cv_outer_folds", "ncvdl", {"summary": r.summary})


def _external_results(S, main):
    d, D = S.design, S.data
    if S.external is None:
        SH.empty(main, "No external cohort yet", "Upload an independent dataset with the same feature columns. "
                 "If it has the outcome, performance is reported with bootstrap CIs; either way, dataset shift is "
                 "measured.")
        return
    r = S.external["result"]
    msgs = r.compatibility.messages()
    SH.section(main, "Feature compatibility")
    for m in msgs:
        main.warning(m)
    if not msgs:
        main.success("All model features present with compatible types.")
    if r.metrics is not None:
        SH.section(main, f"External performance — {S.external['file']}", "Original model, not recalibrated.")
        c1, c2 = main.columns([2, 3])
        c1.dataframe(ci_table(r.metrics), hide_index=True, width="stretch")
        fig_out(c2, VP.ci_forest(r.metrics.loc[~r.metrics["metric"].isin(ERRORS)],
                                     "External metrics with 95% bootstrap CI"))
        p = r.predictions
        yt = np.array([D.classes.index(v) for v in p["observed"]]) if D.classes else p["observed"].to_numpy()
        if d.task == "regression":
            perf_plots(main, "regression", yt, p["prediction"].to_numpy(), None, None, "External", "ext")
        elif all(f"P({c})" in p for c in D.classes):
            pr = p[[f"P({c})" for c in D.classes]].to_numpy()
            perf_plots(main, d.task, yt, np.argmax(pr, 1), pr, D.classes, "External", "ext")
    SH.section(main, "Dataset shift")
    sh = r.shift
    a = sh["domain_auc"]
    main.markdown(f"Domain-classifier AUC = **{a:.2f}** "
                  + ("(cohorts are hard to tell apart)" if a < 0.65 else "(moderate shift)" if a < 0.8
                     else "(strong shift: cohorts are easily distinguished)")
                  + f"; {sh['n_shifted_smd_gt_0.5']} feature(s) with |SMD| > 0.5.")
    if len(sh["features"]):
        fig_out(main, VP.shift_bars(sh["features"]))
    downloads(main, r.predictions.reset_index(drop=True), "external_predictions", "extdl", {"shift": sh["features"]})


# =========================================================================== EXPLAIN
def explain_step(S, module, main, side):
    d, D, sp = S.design, S.data, S.split
    P = SH.panel(side, "explain")
    pipe = _ensure_dev_fit(S, main)
    has_hold = len(sp.test_idx) > 0
    c = SH.card(P, f"{module.key}_xset", "Explanation set")
    opts = (["Holdout"] if has_hold else []) + ["Development"]
    which = SH.picker(c, f"{module.key}_xset", "Samples", opts,
                      help="Holdout samples were not used for fitting. Development samples are in-sample.")
    idx = sp.test_idx if which == "Holdout" else sp.train_idx
    X_eval, y_eval, X_bg = D.X.iloc[idx], D.y[idx], D.X.iloc[sp.train_idx]
    c = SH.card(P, f"{module.key}_xm", "Method")
    method = c.radio("Method", ["SHAP", "Permutation importance", "Model-specific importance",
                                "Partial dependence / ICE"], key=f"{module.key}_xmethod", label_visibility="collapsed")
    main.info(CAVEAT)
    used = model_features_original(pipe, D.categorical)
    main.caption(f"Model: {S.chosen['name']} · fitted on {len(sp.train_idx):,} development samples · "
                 f"{len(used)} of {len(d.features)} original features reach the model.")
    ff = (None, 20, "features")
    if method != "Partial dependence / ICE":
        fc = SH.card(P, f"{module.key}_xff", "Features")
        ff = feature_filter(fc, f"{module.key}_xff", list(D.X.columns), d.covariates, 20)
    if method == "SHAP":
        _shap(S, module, main, P, c, pipe, X_eval, X_bg, which, ff)
    elif method == "Permutation importance":
        _perm(S, module, main, P, c, pipe, X_eval, y_eval, used, which, ff)
    elif method == "Model-specific importance":
        mi = model_specific_importance(pipe, D.categorical)
        if mi is None:
            main.info("This model type has no built-in importance; use SHAP or permutation importance.")
            return
        agg = mi.groupby("feature", as_index=False)["importance"].sum().sort_values("importance", ascending=False)
        if ff[0] is not None:
            agg = agg[agg["feature"].isin(ff[0])]
        fig_out(main, XP.importance_bar(agg, "importance", n=ff[1], title=f"Top {min(ff[1], len(agg))} {ff[2]}: "
                                        f"model-specific importance ({mi.attrs['kind']})",
                                            xlab=mi.attrs["kind"]))
        downloads(main, mi, "model_specific_importance", f"{module.key}_midl")
    else:
        _pdp(S, module, main, P, c, pipe, X_bg, used)


def _shap(S, module, main, P, c, pipe, X_eval, X_bg, which, ff=(None, 20, "features")):
    d, D = S.design, S.data
    n_exp = safe_slider(c, "Samples to explain", min(20, len(X_eval)), min(500, len(X_eval)), 200, 10,
                        key=f"{module.key}_xn")
    n_bg = safe_slider(c, "Background samples", min(20, len(X_bg)), min(300, len(X_bg)), 100, 10,
                       key=f"{module.key}_xbg")
    key = (which, n_exp, n_bg, id(pipe))
    if SH.run_button(P, module, "Compute SHAP", key_suffix="_shap"):
        with main.spinner("Computing SHAP values…"):
            try:
                res = explain_shap(pipe, X_eval, X_bg, d.task, D.classes, n_exp, n_bg, S.val_cfg.random_state)
            except Exception as e:  # noqa: BLE001
                main.error(f"SHAP failed: {type(e).__name__}: {e}"); return
        S.shap = {"key": key, "res": res}
        S.log("SHAP", f"{res.kind} explainer, {len(res.row_index)} samples")
        st.rerun()
    if not S.shap or S.shap["key"][3] != id(pipe):
        SH.empty(main, "SHAP", "Per-sample attributions: how much each feature moved this prediction away from the "
                 "average prediction. Tree and linear models use exact explainers; others use a permutation "
                 "approximation.")
        return
    res = S.shap["res"]
    if S.shap["key"][:3] != key[:3]:
        main.caption("Settings changed: showing the previous result until you compute again.")
    k = positive_output_index(res)
    if len(res.output_names) > 1:
        k = c.selectbox("Output (class)", range(len(res.output_names)), index=k,
                        format_func=lambda i: res.output_names[i], key=f"{module.key}_xout")
    V, names = aggregate_to_original(res.for_output(k), res.feature_names, D.categorical)
    color = res.data.loc[:, [x for x in res.data.columns if to_original(x, D.categorical) == x]]
    out_lab = f"{d.target} (prediction)" if d.task == "regression" else f"{d.target} = {res.output_names[k]}"
    view = SH.picker(main, f"{module.key}_shapview", "View", ["Summary", "Importance", "Dependence", "Single sample"])
    imp = shap_importance(res, k, D.categorical)
    allowed, n_top, glabel = ff
    if allowed is not None:
        keep = [i for i, nm in enumerate(names) if nm in allowed]
        V, names = V[:, keep], [names[i] for i in keep]
        imp = imp[imp["feature"].isin(allowed)].reset_index(drop=True)
    n = min(n_top, len(names))
    if not names:
        main.info(f"None of the {glabel} reach the model."); return
    if view == "Summary":
        fig_out(main, XP.shap_beeswarm(V, names, color, n, f"SHAP summary, top {n} {glabel}: {out_lab} "
                                                           f"({res.output_space})"))
    elif view == "Importance":
        fig_out(main, XP.importance_bar(imp, "mean_abs_shap", n=n, title=f"Top {n} {glabel} by mean |SHAP|",
                                        xlab=f"mean |SHAP| ({res.output_space})"))
    elif view == "Dependence":
        f = main.selectbox("Feature", list(imp["feature"].head(40)), key=f"{module.key}_xdep")
        if f in res.data:
            fig_out(main, XP.dependence(res.data[f].to_numpy(), V[:, names.index(f)], f))
        else:
            main.caption("Categorical feature: see the summary plot or partial dependence.")
    else:
        sid = (S.df.loc[res.row_index, d.sample_id].astype(str).tolist() if d.sample_id
               else [str(i) for i in res.row_index])
        pick = main.selectbox("Sample", range(len(sid)), format_func=lambda i: sid[i], key=f"{module.key}_xloc")
        fig_out(main, XP.waterfall(pd.Series(V[pick], index=names), float(res.base_values[pick, k]),
                                       res.output_space, f"Sample {sid[pick]}: {out_lab}"))
        if d.task != "regression":
            main.caption(f"Observed class: {D.classes[int(D.y[D.index.get_loc(res.row_index[pick])])]}")
    main.caption(f"{res.kind.capitalize()} explainer · {res.output_space} units · {len(res.row_index)} samples. "
                 "One-hot levels are summed into their original feature.")
    downloads(main, imp, "shap_importance", f"{module.key}_shapdl",
              {"shap_values": pd.DataFrame(V, columns=names).assign(row=list(res.row_index))})


def _perm(S, module, main, P, c, pipe, X_eval, y_eval, used, which, ff=(None, 20, "features")):
    d, D = S.design, S.data
    reps = int(c.number_input("Repeats per feature", 2, 50, 10, key=f"{module.key}_pireps"))
    restrict = c.checkbox(f"Only features the model receives ({len(used)})", value=True, key=f"{module.key}_pires")
    if which == "Development":
        c.warning("On training samples this shows what the model memorised; prefer the holdout.")
    pkey = f"{module.key}_perm"
    psig = (id(pipe), which, reps, restrict)

    def apply(job):
        S.perm = job.result
        S.log("Permutation importance", f"{len(S.perm)} features, {reps} repeats ({job.elapsed:.0f} s)")
    pjob = jobs.collect(pkey, apply, psig)
    pbusy = pjob is not None and pjob.running
    if SH.run_button(P, module, "Running in the background…" if pbusy else "Compute permutation importance",
                     disabled=pbusy, key_suffix="_perm"):
        def work(job, feats=tuple(used) if restrict else None):
            r = permutation_importances(pipe, X_eval, y_eval, d.task, len(D.classes or [0, 1]), reps,
                                        S.val_cfg.random_state, features=list(feats) if feats else None,
                                        categorical=D.categorical, progress=job.progress, should_stop=job.stopped)
            return None if job.stopped() else r          # a partial ranking would be misleading
        jobs.start(pkey, "Permutation importance", work, signature=psig)
        st.rerun()
    if pjob is not None and pjob.status == "error":
        main.error(f"Permutation importance did not finish: {pjob.error}")
    with main:
        jobs.monitor(pkey)
    if pbusy:
        return
    if S.perm is None:
        SH.empty(main, "Permutation importance", "The drop in score when one feature is shuffled, holding the "
                 "fitted model fixed. Model-agnostic, but correlated features share importance.")
        return
    pi = S.perm
    main.markdown(f"Baseline {METRIC_LABELS.get(pi.attrs['metric'], pi.attrs['metric'])}: "
                  f"**{pi.attrs['baseline_score']:.3f}**")
    if ff[0] is not None:
        pi = pi[pi["feature"].isin(ff[0])]
    fig_out(main, XP.importance_bar(pi, "importance_mean", "importance_sd", ff[1],
                                    f"Top {min(ff[1], len(pi))} {ff[2]}: permutation importance (mean ± SD)",
                                    "drop in score"))
    downloads(main, pi, "permutation_importance", f"{module.key}_pidl")


def _pdp(S, module, main, P, c, pipe, X_bg, feats):
    d, D = S.design, S.data
    f = c.selectbox("Feature", feats, key=f"{module.key}_pdpf")
    grid = int(c.number_input("Grid points", 5, 50, 20, key=f"{module.key}_pdpg"))
    nice = int(c.number_input("ICE curves", 0, 200, 50, key=f"{module.key}_pdpice"))
    out = -1
    if d.task == "multiclass":
        out = c.selectbox("Class", range(len(D.classes)), format_func=lambda i: D.classes[i], key=f"{module.key}_pdpo")
    if SH.run_button(P, module, "Compute partial dependence", key_suffix="_pdp"):
        with main.spinner("Computing…"):
            try:
                S.analysis = S.analysis or {}
                S.analysis["pdp"] = (f, pdp_ice(pipe, X_bg, f, d.task, D.categorical, out, grid, max(1, nice)))
            except Exception as e:  # noqa: BLE001
                main.error(f"Partial dependence failed: {type(e).__name__}: {e}"); return
    r = (S.analysis or {}).get("pdp")
    if not r or r[0] != f:
        SH.empty(main, "Partial dependence and ICE", "The average prediction as one feature varies (PDP), with "
                 "individual sample curves (ICE). Diverging ICE curves indicate interactions.")
        return
    fig_out(main, XP.pdp_plot(r[1], f))
    main.caption("Computed on development samples. Where features are strongly correlated, PDP evaluates "
                 "unrealistic combinations and can mislead.")


# =========================================================================== PREDICT & DEPLOY
def deploy_step(S, module, main, side):
    d, D, sp = S.design, S.data, S.split
    P = SH.panel(side, "deploy")
    c = SH.card(P, f"{module.key}_fin", "Finalize model",
                "Bundles the pipeline, its metadata and an out-of-distribution detector, signed by this server.")
    has_hold = len(sp.test_idx) > 0
    rows = c.radio("Fit on", ["Development set (the validated model)"] +
                   (["All samples (refit including the holdout)"] if has_hold else []), key=f"{module.key}_finrows")
    if rows.startswith("All"):
        c.caption("Uses all data, as PyCaret's finalize_model does. The refitted model itself has not been "
                  "evaluated; report the holdout or nested-CV estimate of the development model.")
    can = S.chosen is not None
    if SH.run_button(P, module, "Finalize and create bundle", disabled=not can, key_suffix="_finalize"):
        with main.spinner("Fitting and packaging…"):
            idx = sp.train_idx if rows.startswith("Development") else np.arange(len(D.y))
            pipe = (_ensure_dev_fit(S, main) if rows.startswith("Development") else
                    protected(S, main, TK.fit, "Fitting on all samples…", idx=idx, key=S.chosen["key"],
                              params=S.chosen["params"]))
            if pipe is None:
                return
            X = D.X.iloc[idx]
            ood = MM.fit_ood_detector(model_input(pipe, X), S.val_cfg.random_state)
            ho = None
            if S.holdout is not None:
                ho = {r["metric"]: round(float(r["estimate"]), 4) for _, r in S.holdout["metrics"].iterrows()}
            meta = {"module": module.key, "task": d.task, "target": d.target, "classes": D.classes,
                    "model": S.chosen["name"], "model_key": S.chosen["key"],
                    "params": {k: str(v) for k, v in S.chosen["params"].items()},
                    "fitted_on": "development set" if rows.startswith("Development") else "all samples",
                    "n_train": int(len(idx)), "features": list(D.X.columns), "numeric": D.numeric,
                    "categorical": D.categorical, "sample_id": d.sample_id,
                    "profile": MM.training_profile(X, D.numeric, D.categorical),
                    "fingerprint": MM.dataset_fingerprint(X, D.y[idx]),
                    "preprocessing": S.prep_cfg.to_dict(), "validation": S.val_cfg.to_dict(),
                    "holdout_metrics": ho, "holdout_evaluations": S.ledger.n_evaluations if S.ledger else 0}
            S.bundle = MM.make_bundle(pipe, meta, ood)
            S.deploy_model, S.deploy_ood, S.bundle_meta = pipe, ood, MM.read_metadata(S.bundle)
        S.log("Finalize", f"{S.chosen['name']} on {meta['fitted_on']}")
        st.rerun()
    c = SH.card(P, f"{module.key}_load", "Or load a bundle", "Only bundles signed by this server are accepted; "
                "anything else is refused before it is unpickled.")
    up = c.file_uploader("Bundle (.omnix)", type=["omnix", "zip"], key=f"{module.key}_bundleup",
                         label_visibility="collapsed")
    if up is not None and c.button("Load bundle", key=f"{module.key}_loadb", use_container_width=True):
        try:
            m, o, meta = MM.load_bundle(up.getvalue())
            if meta.get("module", module.key) != module.key:
                raise MM.BundleError(f"This bundle is a {meta.get('module')} model, not {module.name.lower()}.")
            S.deploy_model, S.deploy_ood, S.bundle_meta, S.bundle = m, o, meta, up.getvalue()
            S.log("Load bundle", meta.get("model", up.name))
            st.rerun()
        except MM.BundleError as e:
            c.error(str(e))
    if not MM.key_is_persistent():
        P.caption("Bundles reload only while this server keeps running. Set OMNIX_MODEL_KEY on the server to "
                  "reload them after a restart.")

    meta = S.bundle_meta
    if meta is None:
        SH.empty(main, "No deployable model yet", "Finalize the chosen model, or load a bundle created earlier by "
                 "this server. You can then download it and predict new samples with compatibility and "
                 "out-of-distribution flags.")
        return
    SH.section(main, "Model card")
    SH.kpis(main, [("model", meta["model"]), ("fitted on", f"{meta['fitted_on']} (n={meta['n_train']:,})"),
                   ("features", f"{len(meta['features']):,}"), ("created", meta["created"])])
    ho = meta.get("holdout_metrics")
    if ho:
        pm = primary_metric(meta["task"])
        main.caption(f"Holdout {METRIC_LABELS.get(pm, pm)} of the development model: {ho.get(pm, float('nan')):.3f} "
                     f"(holdout evaluated {meta.get('holdout_evaluations', 1)} time(s)).")
    else:
        main.caption("No holdout estimate is attached to this model.")
    with main.expander("Metadata and software versions"):
        st.json({k: v for k, v in meta.items() if k not in ("profile", "features", "numeric", "categorical")})
    if S.bundle is not None:
        main.download_button("Download model bundle", S.bundle, f"{meta['model_key']}_{meta['fingerprint'][:8]}.omnix",
                             "application/zip", key=f"{module.key}_bundledl", type="primary")

    SH.section(main, "Predict new samples", "Same feature columns as training. The outcome column is not needed.")
    new = main.file_uploader("New data", type=["csv", "tsv", "txt", "xlsx"], key=f"{module.key}_newup")
    if new is None:
        return
    try:
        nd = load_table(new.getvalue(), new.name)
    except DataLoadError as e:
        main.error(str(e)); return
    a, b = main.columns(2)
    pct = a.slider("Flag when more unusual than this % of training samples", 90.0, 99.9, 99.0, 0.1,
                   key=f"{module.key}_oodpct")
    rf = b.slider("…or when this share of numeric features is out of range", 0.05, 1.0, 0.2, 0.05,
                  key=f"{module.key}_oodrng")
    try:
        out = MM.predict_new(S.deploy_model, meta, nd, S.deploy_ood, meta.get("sample_id"), pct, rf)
    except Exception as e:  # noqa: BLE001
        main.error(f"Prediction failed: {type(e).__name__}: {e}"); return
    chk = out.attrs["check"]
    if chk["missing"]:
        main.warning(f"{len(chk['missing'])} model feature(s) missing and imputed: {chk['missing'][:10]}")
    for col, lv in chk["unseen_levels"].items():
        main.warning(f"'{col}' has level(s) not seen in training: {lv[:5]}")
    n_flag = int((out["flag"] != "").sum())
    SH.kpis(main, [("samples", f"{len(out):,}"), ("flagged unlike training", f"{n_flag:,}"),
                   ("missing features", str(len(chk["missing"])))])
    if n_flag:
        main.caption("A flag does not mean the prediction is wrong. It means the sample differs from the training "
                     "data, so validated performance may not apply to it.")
    main.dataframe(out.round(4), width="stretch", height=360)
    downloads(main, out.reset_index(drop=True), "predictions", f"{module.key}_preddl")
