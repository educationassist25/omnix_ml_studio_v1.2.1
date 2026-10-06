"""Supervised workflow — step 3 (Compare), step 4 (Create & tune), step 5 (Analyze).

All three use the development rows only (``S.split.train_idx``); the sealed holdout is untouched.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from ..core.explainability import model_specific_importance
from ..core.metrics import LOWER_IS_BETTER, METRIC_LABELS, primary_metric
from ..core.model_comparison import compare_models, registry
from ..visualization import classification_plots as CP
from ..visualization import regression_plots as RP
from ..visualization import validation_plots as VP
from ..visualization import xai_plots as XP
from . import shell as SH
from ..core import tasks as TK
from . import jobs
from .components import download_all_figures, downloads, feature_filter, fig_out, model_picker, param_editor, protected


WIDE_SLOW = {"gbc", "ada", "catboost", "mlp", "rbfsvm", "gpc", "qda", "et", "rf", "xgboost"}


def _metric_cols(lb):
    return [c[:-5] for c in lb.columns if c.endswith("_mean") and lb[c].notna().any()]


def _fmt_params(p: dict) -> dict:
    out = {}
    for k, v in p.items():
        k = k.replace("model__", "")
        if isinstance(v, (float, np.floating)):
            v = float(f"{float(v):.5g}")
        elif isinstance(v, np.integer):
            v = int(v)
        out[k] = v
    return out


def fit_seconds(S, key):
    """Mean seconds per fit for ``key``, measured in the comparison (None if not compared)."""
    r = S.comparison
    if r is None or r.leaderboard.empty or key not in set(r.leaderboard["key"]):
        return None
    row = r.leaderboard.set_index("key").loc[key]
    return float(row["fit_time_s"]) / max(int(r.settings.get("n_splits", 1)), 1)


def fmt_duration(sec):
    if sec is None:
        return "unknown until the model has been compared"
    if sec < 60:
        return "under a minute"
    if sec < 3600:
        return f"about {sec / 60:.0f} min"
    return f"about {sec / 3600:.1f} h"


def grid_size(space: dict):
    """Number of distinct settings when every parameter is a finite list, else None (continuous)."""
    n = 1
    for v in space.values():
        if not isinstance(v, list):
            return None
        n *= len(v)
    return n


# =========================================================================== COMPARE
def compare_step(S, module, main, side):
    d, D, tr = S.design, S.data, S.split.train_idx
    reg = registry(d.task)
    P = SH.panel(side, "compare")
    specs = {k: s for k, s in reg.REGISTRY.items() if k != "dummy"}
    c = SH.card(P, f"{module.key}_algos", "Algorithms",
                f"{len(specs)} available. The dummy baseline is always included as a reference.")
    badge = (lambda s: "" if s.proba else "no probabilities") if d.task != "regression" else None
    keys = model_picker(c, f"{module.key}_pick", specs, reg.FAMILIES, [k for k in reg.DEFAULT if k != "dummy"],
                        badge=badge, columns=1)
    c.caption(f"{len(keys)} selected")
    c = SH.card(P, f"{module.key}_rank", "Budget and computing")
    pm = primary_metric(d.task)
    from ..core.runner import available_cpus, available_memory_mb, plan_workers, resolve_jobs
    lim = c.select_slider("Time limit per model", [1, 2, 5, 10, 20, 30, 60, 0], value=10,
                          format_func=lambda m: "none" if m == 0 else f"{m} min", key=f"{module.key}_permodel",
                          help="A model that takes longer is stopped and reported; the others carry on.")
    budget = c.number_input("Total time budget (minutes, 0 = none)", 0, 600, 0, 5, key=f"{module.key}_budgetmin",
                            help="Models not started when the budget runs out are skipped.")
    n_w, mem_w = plan_workers(resolve_jobs(S.val_cfg.n_jobs))
    c.caption(f"Runs {n_w} model(s) at a time on {available_cpus()} CPU core(s), each in its own protected process "
              f"with up to {mem_w / 1024:.1f} GB of memory ({available_memory_mb() / 1024:.1f} GB free).")
    p_now = len(d.features) if not S.prep_cfg.univariate_k else min(S.prep_cfg.univariate_k, len(d.features))
    if p_now > 2000:
        c.warning(f"{p_now:,} features reach the models (no univariate pre-filter). Boosting, AdaBoost, CatBoost, "
                  "MLP and kernel models will be slow. A pre-filter in Setup (top 1,000, refitted in every fold) "
                  "speeds everything up.")
    heavy = [specs[k].name for k in keys if specs[k].speed == "slow" or (p_now > 2000 and k in WIDE_SLOW)]
    if heavy:
        c.caption("Slow on these data: " + ", ".join(heavy))
    jkey = f"{module.key}_compare"
    sig = (id(S.data), repr(S.prep_cfg), repr(S.val_cfg), S.split.description)

    def apply(job):
        r = job.result
        S.invalidate("models")
        S.comparison = r
        if len(r.leaderboard):
            b = r.leaderboard.iloc[0]
            S.log("Compare", f"{len(r.leaderboard)} models in {job.elapsed / 60:.1f} min; best {b['Model']} "
                             f"{METRIC_LABELS[pm]} = {b[pm + '_mean']:.3f}" + (" (stopped early)" if job.status == "stopped" else ""))

    job = jobs.collect(jkey, apply, sig)
    busy = job is not None and job.running
    if SH.run_button(P, module, "Running in the background…" if busy else f"Compare {len(keys)} models",
                     disabled=not keys or busy, key_suffix="_compare"):
        names = {k: reg.REGISTRY[k].name for k in ["dummy"] + keys}

        def work(job):
            return compare_models(D, d.task, ["dummy"] + keys, S.prep_cfg, S.val_cfg, train_idx=tr,
                                  time_budget_s=budget * 60 or None, isolate=True, per_model_timeout_s=lim * 60 or None,
                                  workers=n_w, mem_limit_mb=mem_w,
                                  progress=lambda i, n, k: job.progress(i, n, names.get(k, k)),
                                  should_stop=job.stopped, on_result=job.add)
        jobs.start(jkey, f"Comparing {len(keys) + 1} models", work, signature=sig)
        st.rerun()
    if job is not None and job.status == "error":
        main.error(f"The comparison did not finish: {job.error}")
    with main:
        jobs.monitor(jkey, lambda j: jobs.partial_table(j, pm, METRIC_LABELS[pm], pm in LOWER_IS_BETTER))
    if busy:
        return

    r = S.comparison
    SH.kpis(main, [("development samples", f"{len(tr):,}"), ("features", f"{len(d.features):,}"),
                   ("validation", f"{S.val_cfg.n_folds}-fold" + (f" × {S.val_cfg.n_repeats}" if S.val_cfg.n_repeats > 1 else "")),
                   ("holdout", "sealed" if len(S.split.test_idx) else "none")])
    if r is None:
        SH.empty(main, "No comparison yet", "Choose algorithms in the panel and press Compare. Every model is "
                 "cross-validated on identical folds of the development set, with preprocessing refitted inside "
                 "each training fold.")
        return
    lb = r.leaderboard
    if lb.empty:
        msgs = list(dict.fromkeys(r.failures.values()))
        main.error("No model could be trained. The same error affected every model, including the dummy baseline, "
                   "so the cause is in the data or preprocessing:\n\n" + "\n\n".join(f"- {m}" for m in msgs[:3]))
        return
    mets = _metric_cols(lb)
    a, b = main.columns([2, 1], vertical_alignment="bottom")
    sort_m = a.selectbox("Sort by", mets, index=mets.index(pm) if pm in mets else 0,
                         format_func=lambda m: METRIC_LABELS.get(m, m), key=f"{module.key}_sortby")
    show_sd = b.toggle("Show ± SD", value=False, key=f"{module.key}_showsd")
    lbs = lb.sort_values(f"{sort_m}_mean", ascending=sort_m in LOWER_IS_BETTER, na_position="last")
    SH.section(main, "Leaderboard", f"{r.settings['cv']}. Mean across folds of the development set; best value "
                                    "in each column is highlighted. Internal validation, not holdout performance.")
    tab = pd.DataFrame({"Model": lbs["Model"].values})
    for m in mets:
        tab[METRIC_LABELS.get(m, m)] = pd.to_numeric(lbs[f"{m}_mean"], errors="coerce").values
    tab["Time (s)"] = lbs["fit_time_s"].round(2).values
    if show_sd:
        disp = tab.copy()
        for m in mets:
            lab = METRIC_LABELS.get(m, m)
            disp[lab] = [("—" if not np.isfinite(x) else f"{x:.3f} ± {s:.3f}" if np.isfinite(s) else f"{x:.3f}")
                         for x, s in zip(lbs[f"{m}_mean"], lbs[f"{m}_sd"])]
        main.dataframe(disp, hide_index=True, width="stretch")
    else:
        sty = tab.style.format({**{METRIC_LABELS.get(m, m): "{:.3f}" for m in mets}, "Time (s)": "{:.2f}"}, na_rep="—")
        for m in mets:
            lab = METRIC_LABELS.get(m, m)
            fn = sty.highlight_min if m in LOWER_IS_BETTER else sty.highlight_max
            sty = fn(subset=[lab], props=f"background-color:{module.tint}; font-weight:600; color:{module.color}")
        main.dataframe(sty, hide_index=True, width="stretch", height=min(38 + 35 * len(tab), 640))
    if d.task != "regression" and not lbs["probabilities"].all():
        main.caption("— = not available: models without probability estimates have no AUC, PR-AUC, Brier or log "
                     "loss in multiclass tasks.")
    if "beats_baseline" in lb and not lb.loc[lb["key"] != "dummy", "beats_baseline"].any():
        main.error("No model beats the dummy baseline: there may be no learnable signal in these features.")
    if d.task == "binary" and min(pd.Series(D.y[tr]).value_counts(normalize=True)) < 0.2:
        main.info("Imbalanced outcome: accuracy can look high for a model that ignores the minority class. "
                  "Prefer balanced accuracy, MCC or PR-AUC.")
    if r.failures:
        with main.expander(f"{len(r.failures)} model(s) failed or were skipped"):
            for k, msg in r.failures.items():
                st.markdown(f"**{reg.REGISTRY[k].name if k in reg.REGISTRY else k}**: {msg}")
    c1, c2 = main.columns(2)
    fig_out(c1, VP.leaderboard_bars(lb, sort_m), key=f"{module.key}_lbbars")
    fig_out(c2, VP.fold_strip(r.folds, lb, sort_m), key=f"{module.key}_lbfolds")
    downloads(main, lb, "model_comparison", f"{module.key}_lbdl",
              {"settings": pd.DataFrame([{k: str(v) for k, v in r.settings.items()}])})
    best = lbs.loc[lbs["key"] != "dummy"].iloc[0] if (lbs["key"] != "dummy").any() else None
    if best is not None:
        if main.button(f"Create and tune {best['Model']}", key=f"{module.key}_tobest", type="primary"):
            st.session_state[f"{module.key}_cm_key"] = best["key"]
            SH.go_to(module, "create")
            st.rerun()


# =========================================================================== CREATE & TUNE
def create_step(S, module, main, side):
    d = S.design
    reg = registry(d.task)
    S.created = S.created or {}
    P = SH.panel(side, "create")
    keys = [k for k in reg.REGISTRY if k != "dummy"]
    if S.comparison is not None and len(S.comparison.leaderboard):
        ranked = [k for k in S.comparison.leaderboard["key"] if k != "dummy"]
        keys = ranked + [k for k in keys if k not in ranked]
    ck = f"{module.key}_cm_key"
    if st.session_state.get(ck) not in keys:
        st.session_state[ck] = keys[0]
    lb = S.comparison.leaderboard.set_index("key") if S.comparison is not None and len(S.comparison.leaderboard) else None
    pm = primary_metric(d.task)

    def lab(k):
        s = reg.REGISTRY[k].name
        if lb is not None and k in lb.index and np.isfinite(lb.loc[k, f"{pm}_mean"]):
            s += f"  ·  CV {lb.loc[k, f'{pm}_mean']:.3f}"
        return s

    c = SH.card(P, f"{module.key}_cm", "Model", "Ordered by the comparison leaderboard.")
    key = c.selectbox("Algorithm", keys, format_func=lab, key=ck)
    spec = reg.REGISTRY[key]
    if spec.notes:
        c.caption(spec.notes)
    c = SH.card(P, f"{module.key}_hp", "Hyperparameters")
    modes = ["Defaults", "Custom", "Random search"] if spec.space else ["Defaults"]
    mode = SH.picker(c, f"{module.key}_hpmode_{key}", "Mode", modes)
    params, n_iter, opt = {}, 0, pm
    if mode == "Custom":
        params = param_editor(c, f"{module.key}_pe_{key}", spec.factory(S.val_cfg.random_state), spec.space)
        c.caption(f"{len(params)} changed from the default" if params else "Edit any value to override the default.")
    elif mode == "Random search":
        n_iter = int(c.slider("Iterations", 10, 500, 100, 10, key=f"{module.key}_iter",
                              help="Number of hyperparameter settings tried. 100 or more is recommended."))
        n_cv = S.val_cfg.n_folds
        per = fit_seconds(S, key)
        g = grid_size(spec.space)
        eff = min(n_iter, g) if g else n_iter
        c.caption(f"{eff:,} settings × {n_cv} folds = {eff * n_cv:,} fits; "
                  f"{fmt_duration(per * eff * n_cv * 1.3 if per else None)}.")
        if g and n_iter > g:
            c.info(f"This search space has only {g} distinct settings, so the search stops after {g}.")
        mets = [m for m in (lb.columns if lb is not None else []) if m.endswith("_mean")]
        opts = [m[:-5] for m in mets if lb[m].notna().any() and m[:-5] not in LOWER_IS_BETTER] or [pm]
        opt = c.selectbox("Optimise", opts, index=opts.index(pm) if pm in opts else 0,
                          format_func=lambda m: METRIC_LABELS.get(m, m), key=f"{module.key}_opt")
        with c.expander(f"Search space ({len(spec.space)} parameters)"):
            st.dataframe(pd.DataFrame({"parameter": [k.replace("model__", "") for k in spec.space],
                                       "values": [str(v)[:80] if isinstance(v, list) else type(v).__name__
                                                  for v in spec.space.values()]}),
                         hide_index=True, width="stretch")
        c.caption("Tuned on the development set. The best of many settings is optimistic; the holdout or nested "
                  "CV gives the unbiased estimate.")
    verb = {"Defaults": "Create model", "Custom": "Create model", "Random search": f"Tune ({n_iter} iterations)"}[mode]
    tkey = f"{module.key}_tune"
    sig = (id(S.data), repr(S.prep_cfg), repr(S.val_cfg), S.split.description)

    def register(cm, how, t=None):
        if t is not None:
            S.tuning[cm.key] = t
        name = f"{cm.name} — {how}"
        S.created[name] = cm
        st.session_state[f"{module.key}_cm_view"] = name
        S.log("Create", f"{name}: {METRIC_LABELS.get(pm, pm)} {cm.summary.get(pm, (np.nan,))[0]:.3f}")

    tjob = jobs.collect(tkey, lambda j: register(j.result[1], j.result[2], j.result[0]), sig)
    tbusy = tjob is not None and tjob.running
    if SH.run_button(P, module, "Tuning in the background…" if tbusy else verb, disabled=tbusy, key_suffix="_create"):
        if mode == "Random search":
            payload = {"data": S.data, "idx": S.split.train_idx, "task": d.task, "prep": S.prep_cfg, "val": S.val_cfg}

            def work(job, key=key, n_iter=n_iter, opt=opt, name=spec.name):
                from ..core.runner import run_isolated
                job.progress(0, 2, f"searching {n_iter} settings of {name}")
                r = run_isolated(payload, TK.tune, [("tune", {"key": key, "n_iter": n_iter, "metric": opt})], workers=1,
                                 should_stop=job.stopped)["tune"]
                if job.stopped():
                    return None
                if r.status != "ok":
                    raise RuntimeError(r.message.split("\n")[0])
                t = r.value
                how = f"tuned ({t.n_iter} iterations, {METRIC_LABELS.get(t.metric, t.metric)})"
                job.progress(1, 2, "cross-validating the best setting on the comparison folds")
                r2 = run_isolated(payload, TK.create, [("create", {"key": key, "params": dict(t.best_params), "how": how})],
                                  workers=1, should_stop=job.stopped)["create"]
                if r2.status != "ok":
                    raise RuntimeError(r2.message.split("\n")[0])
                job.progress(2, 2, "done")
                return t, r2.value, how
            jobs.start(tkey, f"Tuning {spec.name}", work, signature=sig)
            st.rerun()
        how = mode.lower()
        cm = protected(S, main, TK.create, f"Cross-validating {spec.name}…", key=key, params=params, how=how)
        if cm is None:
            return
        register(cm, how)
        st.rerun()
    if tjob is not None and tjob.status == "error":
        main.error(f"Tuning did not finish: {tjob.error}")
    with main:
        jobs.monitor(tkey)

    if not S.created:
        SH.empty(main, "No model created yet", "Pick an algorithm and a hyperparameter mode in the panel. Created "
                 "and tuned models are cross-validated on the same folds as the comparison, so their scores are "
                 "directly comparable.")
        return
    SH.section(main, "Created models", "Same folds as the comparison. Choose one as the final model before you "
                                       "open the holdout.")
    rows = []
    for n, cm in S.created.items():
        r = {"Model": n}
        for m, (mu, _sd) in cm.summary.items():
            r[METRIC_LABELS.get(m, m)] = mu
        r["final"] = "✓" if S.chosen and S.chosen["name"] == n else ""
        rows.append(r)
    main.dataframe(pd.DataFrame(rows).round(4), hide_index=True, width="stretch")
    names = list(S.created)
    vk = f"{module.key}_cm_view"
    if st.session_state.get(vk) not in names:
        st.session_state[vk] = names[-1]
    a, b = main.columns([3, 1], vertical_alignment="bottom")
    view = a.selectbox("Inspect", names, key=vk)
    cm = S.created[view]
    is_final = bool(S.chosen and S.chosen["name"] == view)
    if b.button("Final model ✓" if is_final else "Set as final model", key=f"{module.key}_setfinal",
                type="secondary" if is_final else "primary", use_container_width=True, disabled=is_final):
        S.chosen = {"key": cm.key, "params": cm.params, "source": cm.how, "name": view}
        S.final_model, S.holdout, S.external, S.shap, S.perm = None, None, None, None, None
        S.oof, S.analysis, S.bundle, S.bundle_meta = None, None, None, None
        S.log("Final model", view)
        SH.advance(module, "analyze")
        st.rerun()
    c1, c2 = main.columns([3, 2])
    c1.markdown("**Per-fold scores**")
    c1.dataframe(cm.folds.round(4), width="stretch")
    c2.markdown("**Hyperparameters**")
    fp = _fmt_params(cm.params)
    if fp:
        c2.dataframe(pd.DataFrame({"parameter": list(fp), "value": [str(v) for v in fp.values()]}),
                     hide_index=True, width="stretch")
    else:
        c2.caption("Library defaults.")
    t = S.tuning.get(cm.key)
    if t is not None and cm.how.startswith("tuned"):
        with main.expander("Search history (top 20 settings)"):
            st.dataframe(t.cv_results.head(20).astype(str), hide_index=True, width="stretch")
    if S.chosen:
        main.success(f"Final model: {S.chosen['name']}. Analyze it on out-of-fold predictions, then validate.")


# =========================================================================== ANALYZE
CLS_PLOTS = ["ROC curve", "Precision–recall", "Confusion matrix", "Threshold", "Calibration", "Gain and lift",
             "Class report", "Learning curve", "Feature importance", "Hyperparameters"]
REG_PLOTS = ["Residuals", "Prediction error", "Q–Q plot", "Learning curve", "Feature importance", "Hyperparameters"]
BINARY_ONLY = ("Threshold", "Calibration", "Gain and lift")
NEEDS_SCORES = ("ROC curve", "Precision–recall", "Threshold", "Calibration", "Gain and lift")


def _comparison_card(P, module, d, D, pr):
    """Which comparison the classification plots show. Returns a dict describing the view."""
    cls = list(D.classes)
    c = SH.card(P, f"{module.key}_cmpview", "Comparison", "Choose which classes the curves compare.")
    if d.task == "binary":
        pos = c.selectbox("Positive class", [cls[1], cls[0]], key=f"{module.key}_an_pos",
                          help=f"Set to '{cls[1]}' in the Data step. Switching it re-orients precision–recall, "
                               "threshold and gain curves; ROC-AUC does not change.")
        return {"mode": "binary", "flip": pos != cls[1], "label": f"{pos} vs {cls[0] if pos == cls[1] else cls[1]}"}
    modes = ["One vs rest", "Class A vs class B"]
    mode = SH.picker(c, f"{module.key}_an_mode", "Mode", modes,
                     help="One vs rest: each class against all others. A vs B: only samples of the two classes.")
    if mode == "One vs rest":
        show = c.multiselect("Classes to show", cls, default=cls, key=f"{module.key}_an_show")
        one = c.selectbox("Single-class plots (threshold, calibration, gain) for", cls, key=f"{module.key}_an_one")
        return {"mode": "ovr", "show": [cls.index(x) for x in show] or list(range(len(cls))), "one": one,
                "label": f"{one} vs rest"}
    a, b = c.columns(2)
    ca = a.selectbox("Class A (positive)", cls, index=0, key=f"{module.key}_an_a")
    cb = b.selectbox("Class B", [x for x in cls if x != ca], key=f"{module.key}_an_b")
    return {"mode": "pair", "a": ca, "b": cb, "label": f"{ca} vs {cb}"}


def _binary_view(view, y, pr, classes):
    """(y01, P(n,2), labels) for binary-type plots under the chosen comparison."""
    from ..core import comparisons as CMP
    if view["mode"] == "binary":
        return CMP.flip_binary(y, pr, classes) if view["flip"] else (np.asarray(y), pr, list(classes))
    if view["mode"] == "ovr":
        yb, P2, lab, _ = CMP.one_vs_rest(y, pr, classes, view["one"])
        return yb, P2, lab
    yb, P2, lab, _ = CMP.pairwise(y, pr, classes, view["a"], view["b"])
    return yb, P2, lab


def _class_figures(plot, view, y, yp, pr, cls, norm, nbins, calibrated):
    """List of (figure, caption) for one classification plot under the chosen comparison."""
    out = []
    if plot in NEEDS_SCORES and pr is None:
        return out
    if plot in ("ROC curve", "Precision–recall"):
        fn = CP.roc if plot == "ROC curve" else CP.pr
        if view["mode"] == "ovr":
            out.append((fn(y, pr, cls, f"{plot} (out-of-fold): one vs rest", only=view["show"]), ""))
        else:
            yb, P2, lab = _binary_view(view, y, pr, cls)
            out.append((fn(yb, P2, lab, f"{plot} (out-of-fold): {view['label']}"),
                        "Only samples of the two classes are used." if view["mode"] == "pair" else ""))
    elif plot == "Confusion matrix":
        out.append((CP.confusion(y, yp, cls, "Confusion matrix (out-of-fold)", normalize=norm), ""))
    elif plot == "Class report":
        out.append((CP.class_report(y, yp, cls), ""))
    else:
        yb, P2, lab = _binary_view(view, y, pr, cls)
        if plot == "Threshold":
            out.append((CP.threshold_analysis(yb, P2, f"Threshold analysis: {view['label']}"),
                        "Choosing a threshold here is itself a modelling decision; confirm it on the holdout."))
        elif plot == "Calibration":
            out.append((CP.calibration(yb, P2, nbins, f"Calibration: {view['label']}"),
                        "" if calibrated and view["mode"] != "pair" else
                        "Scores are not calibrated probabilities (no predict_proba, or a pairwise ratio)."))
        elif plot == "Gain and lift":
            out.append((CP.gain_lift(yb, P2, f"Cumulative gain and lift: {view['label']}"), ""))
    return out


def _importance_figure(S, D, pipe, allowed, n, glabel, spec):
    mi = model_specific_importance(pipe, D.categorical)
    if mi is None:
        return None, None, f"{spec.name} has no built-in importance. Use SHAP or permutation importance in Explain."
    agg = mi.groupby("feature", as_index=False)["importance"].sum().sort_values("importance", ascending=False)
    if allowed is not None:
        agg = agg[agg["feature"].isin(allowed)]
    agg = agg.reset_index(drop=True)
    agg.insert(0, "rank", np.arange(1, len(agg) + 1))
    fig = XP.importance_bar(agg, "importance", n=n, title=f"Top {min(n, len(agg))} {glabel} by model-specific "
                            f"importance ({mi.attrs['kind']})", xlab=mi.attrs["kind"])
    fig.update_layout(height=max(320, 90 + 24 * min(n, len(agg))))
    return fig, agg, None


def analyze_step(S, module, main, side):
    d, D, tr, ch = S.design, S.data, S.split.train_idx, S.chosen
    reg = registry(d.task)
    spec = reg.REGISTRY[ch["key"]]
    P = SH.panel(side, "analyze")
    plots = list(REG_PLOTS) if d.task == "regression" else list(CLS_PLOTS)
    S.analysis = S.analysis or {}
    an, oof_key = S.analysis, ch["name"]
    if S.oof is None or S.oof[0] != oof_key:
        with main.spinner("Computing out-of-fold predictions…"):
            try:
                res = protected(S, main, TK.oof, "Computing out-of-fold predictions…", key=ch["key"],
                                params=ch["params"])
                if res is None:
                    return
                S.oof = (oof_key, *res)
            except Exception as e:  # noqa: BLE001
                main.error(f"Out-of-fold prediction failed: {e}"); return
    _, y, yp, pr = S.oof

    c = SH.card(P, f"{module.key}_plot", "Plots")
    show_all = SH.picker(c, f"{module.key}_an_all", "Show", ["One plot", "All plots"]) == "All plots"
    plot = None if show_all else c.radio("Plot", plots, key=f"{module.key}_plotsel", label_visibility="collapsed")
    view = _comparison_card(P, module, d, D, pr) if d.task != "regression" else None
    norm, nbins = "row", 10
    if d.task != "regression" and (show_all or plot == "Confusion matrix"):
        norm = {"Row % (recall)": "row", "Column % (precision)": "column", "Counts": "none"}[
            c.selectbox("Confusion matrix", ["Row % (recall)", "Column % (precision)", "Counts"],
                        key=f"{module.key}_an_norm")]
    if d.task != "regression" and (show_all or plot == "Calibration"):
        nbins = int(c.slider("Calibration bins", 5, 20, 10, key=f"{module.key}_calbins"))
    allowed, n, glabel = None, 20, "features"
    if show_all or plot == "Feature importance":
        fc = SH.card(P, f"{module.key}_feat", "Feature importance")
        allowed, n, glabel = feature_filter(fc, f"{module.key}_fi", list(D.X.columns), d.covariates, 20)
    if show_all or plot == "Learning curve":
        n_pts = int(c.slider("Learning-curve points", 3, 10, 6, key=f"{module.key}_lcpts"))
    run_label = "Generate all plots" if show_all else ("Compute learning curve" if plot == "Learning curve" else None)
    if run_label and SH.run_button(P, module, run_label, key_suffix="_lc"):
        with main.spinner("Fitting on increasing training sizes…"):
            try:
                res = protected(S, main, TK.learning_curve, "Fitting on increasing training sizes…", key=ch["key"],
                                params=ch["params"], n_points=n_pts)
                if res is not None:
                    an["lc"] = (oof_key, res)
            except Exception as e:  # noqa: BLE001
                main.error(f"Learning curve failed: {e}")
        an["all_ready"] = oof_key if show_all else an.get("all_ready")
    if (show_all or plot == "Feature importance") and an.get("pipe_for") != oof_key:
        with main.spinner("Fitting on the development set…"):
            pipe = protected(S, main, TK.fit, "Fitting on the development set…", key=ch["key"], params=ch["params"])
            if pipe is not None:
                an["pipe"], an["pipe_for"] = pipe, oof_key

    cmp_txt = ""
    if view:
        cmp_txt = (f" · comparison: **one vs rest** (single-class plots: {view['label']})" if view["mode"] == "ovr"
                   else f" · comparison: **{view['label']}**")
    main.markdown(f"**{ch['name']}** · out-of-fold predictions on {len(tr):,} development samples" + cmp_txt)

    def figures_for(p):
        """[(fig, caption)] or a string message."""
        if p == "Hyperparameters":
            return "table"
        if p == "Learning curve":
            lc = an.get("lc")
            if not lc or lc[0] != oof_key:
                return "Press the button in the panel to compute the learning curve."
            sizes, a, b, m = lc[1]
            return [(CP.learning_curve_fig(sizes, a, b, METRIC_LABELS.get(m, m)),
                     "A persistent gap between curves suggests over-fitting; two low, converged curves suggest "
                     "under-fitting.")]
        if p == "Feature importance":
            fig, _, msg = _importance_figure(S, D, an["pipe"], allowed, n, glabel, spec)
            return msg if fig is None else [(fig, "Importance is not significance. Coefficients are on the scaled "
                                                  "input; impurity importance favours continuous features.")]
        if d.task == "regression":
            return [({"Residuals": RP.residuals, "Prediction error": RP.observed_vs_predicted, "Q–Q plot": RP.qq}[p](
                y, yp, *(() if p == "Q–Q plot" else (f"{p} (out-of-fold)",))), "")]
        if p in NEEDS_SCORES and pr is None:
            return f"{spec.name} gives no probability or decision scores for this task, so {p} is not available."
        return _class_figures(p, view, y, yp, pr, D.classes, norm, nbins, bool(spec.proba))

    if not show_all:
        res = figures_for(plot)
        if res == "table":
            _hyper_table(main, spec, ch, S)
        elif isinstance(res, str):
            main.info(res)
        else:
            for fig, cap in res:
                fig_out(main, fig, key=f"{module.key}_an_{plot}")
                if cap:
                    main.caption(cap)
        if plot == "Feature importance" and an.get("pipe") is not None:
            _, agg, _ = _importance_figure(S, D, an["pipe"], allowed, n, glabel, spec)
            if agg is not None:
                with main.expander(f"Full ranking ({len(agg):,} {glabel})"):
                    st.dataframe(agg.round(5), hide_index=True, width="stretch", height=320)
                downloads(main, agg, f"feature_importance_{glabel.split()[0]}", f"{module.key}_fidl")
        return

    # ---- gallery
    gallery = [p for p in plots if p != "Hyperparameters"]
    items, notes = [], []
    for p in gallery:
        res = figures_for(p)
        if isinstance(res, str):
            notes.append(f"{p}: {res}")
        else:
            items += [(p, f, cap) for f, cap in res]
    SH.section(main, f"All plots ({len(items)})", "Each plot has its own download menu; download every plot at "
                                                  "once from the menu here.")
    download_all_figures(main, f"{module.key}_analysis", [(f"{p}_{i}", f) for i, (p, f, _) in enumerate(items)],
                         label="Download all plots")
    st.session_state["_page_zip_done"] = True
    for i in range(0, len(items), 2):
        cols = main.columns(2)
        for col, (p, f, cap) in zip(cols, items[i:i + 2]):
            fig_out(col, f, key=f"{module.key}_gal_{i}_{p}", name=p)
            if cap:
                col.caption(cap)
    for m in notes:
        main.caption(m)
    with main.expander("Hyperparameters"):
        _hyper_table(st, spec, ch, S)


def _hyper_table(parent, spec, ch, S):
    pipe_params = spec.factory(S.val_cfg.random_state).get_params()
    over = _fmt_params(ch["params"])
    rows = [{"parameter": k, "value": str(over.get(k, v)), "changed": "✓" if k in over else ""}
            for k, v in pipe_params.items() if not callable(v) and not hasattr(v, "get_params")]
    parent.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=420)
