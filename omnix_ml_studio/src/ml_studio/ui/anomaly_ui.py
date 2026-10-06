"""Anomaly detection: Compare, Create, Analyze, Validate, Explain, Score & deploy."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from ..core import anomaly as AN
from ..core import clustering as CL
from ..core import model_manager as MM
from ..core.data_validation import DataLoadError, load_table
from ..visualization import unsup_plots as UP
from . import jobs
from . import shell as SH
from .components import downloads, fig_out, model_picker
from .unsup_common import clear, spec_param_editor

FAMILIES = ["Ensemble", "Proximity", "Statistical", "Linear", "Kernel"]
AL = AN.ANOMALY_LABELS
REF_METRICS = ["roc_auc", "average_precision", "precision_at_n", "precision_flagged", "recall_flagged"]


def _contam(S):
    return S.u["setup"].get("contamination", 0.05)


# =========================================================================== COMPARE
def compare_step(S, module, main, side):
    ud, Z = S.u["ud"], S.u["Z"]
    P = SH.panel(side, "compare")
    specs = dict(AN.REGISTRY)
    c = SH.card(P, f"{module.key}_algos", "Detectors", f"{len(specs)} available (PyOD).")
    keys = model_picker(c, f"{module.key}_pick", specs, FAMILIES, AN.DEFAULT,
                        badge=lambda s: "batch only" if s.transductive else "", columns=1)
    c.caption("Batch-only detectors score a dataset relative to itself and cannot score new samples later.")
    SH.kpis(main, [("samples", f"{len(Z):,}"), ("model dimensions", str(Z.shape[1])),
                   ("contamination", f"{_contam(S):.0%}"), ("reference", ud.reference_name or "none")])
    jkey = f"{module.key}_compare"

    def apply(job):
        clear(S, "compare")
        S.u["cmp"] = job.result
        S.log("Compare", f"{len(job.result.table)} detectors")
    job = jobs.collect(jkey, apply, id(Z))
    busy = job is not None and job.running
    if SH.run_button(P, module, "Running in the background…" if busy else f"Compare {len(keys)} detectors",
                     disabled=not keys or busy, key_suffix="_compare"):
        def work(job, keys=tuple(keys), contam=_contam(S)):
            return jobs.run_stoppable(job, AN.compare_detectors, Z, list(keys), contam, ud.reference,
                                      progress=jobs.stoppable_progress(job))
        jobs.start(jkey, f"Comparing {len(keys)} detectors", work, signature=id(Z))
        st.rerun()
    if job is not None and job.status == "error":
        main.error(f"The comparison did not finish: {job.error}")
    with main:
        jobs.monitor(jkey)
    if busy:
        return
    r = S.u.get("cmp")
    if r is None:
        SH.empty(main, "No comparison yet", "Every detector is fitted to the same preprocessed data and flags the "
                 "top contamination fraction. Without a reference label there is no accuracy to rank by, so the "
                 "comparison shows how far the detectors agree.")
        return
    if r.table.empty:
        main.error("Every detector failed: " + "; ".join(list(r.failures.values())[:3])); return
    names = dict(zip(r.table["key"], r.table["Model"]))
    tab = r.table.drop(columns=["key"]).rename(columns={**AL, "family": "Family"})
    tab["consensus overlap"] = [float(((r.flags[k] == 1) & (r.consensus.rank(ascending=False) <= r.flags[k].sum())).sum()
                                      / max(r.flags[k].sum(), 1)) for k in r.table["key"]]
    SH.section(main, "Leaderboard", "Ranked by PR-AUC against the reference label." if ud.reference is not None else
               "No reference label: 'consensus overlap' is the share of a detector's flags that are also among "
               "the top-ranked samples by mean rank across all detectors.")
    fmt = {AL[m]: "{:.3f}" for m in REF_METRICS if AL[m] in tab}
    fmt.update({"Fit time (s)": "{:.2f}", "consensus overlap": "{:.0%}"})
    sty = tab.style.format(fmt, na_rep="—")
    for col in [AL[m] for m in REF_METRICS if AL[m] in tab] + ["consensus overlap"]:
        sty = sty.highlight_max(subset=[col], props=f"background-color:{module.tint}; font-weight:600; color:{module.color}")
    main.dataframe(sty, hide_index=True, width="stretch")
    if r.failures:
        with main.expander(f"{len(r.failures)} detector(s) failed"):
            for k, msg in r.failures.items():
                st.markdown(f"**{AN.get(k).name}**: {msg}")
    if r.scores.shape[1] > 1:
        v = SH.picker(main, f"{module.key}_agview", "Agreement", ["Score rank correlation", "Flag overlap (Jaccard)"])
        M = r.agreement if v.startswith("Score") else r.flag_jaccard
        fig_out(main, UP.agreement_heatmap(M, names, v))
    downloads(main, r.table, "detector_comparison", f"{module.key}_lbdl",
              {"scores": r.scores.assign(consensus=r.consensus).reset_index(drop=True)})
    best = r.table.iloc[0]
    if main.button(f"Create {best['Model']}", key=f"{module.key}_tobest", type="primary"):
        st.session_state[f"{module.key}_cm_key"] = best["key"]
        SH.go_to(module, "create")
        st.rerun()


# =========================================================================== CREATE
def create_step(S, module, main, side):
    ud, Z = S.u["ud"], S.u["Z"]
    models = S.u.setdefault("models", {})
    P = SH.panel(side, "create")
    keys = list(AN.REGISTRY)
    if S.u.get("cmp") is not None and len(S.u["cmp"].table):
        ranked = list(S.u["cmp"].table["key"])
        keys = ranked + [k for k in keys if k not in ranked]
    ck = f"{module.key}_cm_key"
    if st.session_state.get(ck) not in keys:
        st.session_state[ck] = keys[0]
    c = SH.card(P, f"{module.key}_cm", "Detector")
    key = c.selectbox("Detector", keys, format_func=lambda k: AN.get(k).name, key=ck, label_visibility="collapsed")
    spec = AN.get(key)
    if spec.notes:
        c.caption(spec.notes)
    c = SH.card(P, f"{module.key}_hp", "Parameters")
    params = spec_param_editor(c, f"{module.key}_pe_{key}", spec.params)
    contam = float(c.slider("Contamination", 0.01, 0.3, float(_contam(S)), 0.01, key=f"{module.key}_cmcont"))
    if SH.run_button(P, module, "Create model", key_suffix="_create"):
        try:
            with main.spinner(f"Fitting {spec.name}…"):
                model, sc, fl = AN.fit_detector(Z, key, contam, params)
        except Exception as e:  # noqa: BLE001
            main.error(f"{spec.name} failed: {type(e).__name__}: {e}"); return
        desc = ", ".join(f"{k}={v}" for k, v in params.items())
        name = f"{spec.name} ({contam:.0%}" + (f", {desc})" if desc else ")")
        met = {"n_flagged": int(fl.sum())}
        if ud.reference is not None:
            met.update(AN.label_metrics(sc, fl, ud.reference))
        models[name] = {"key": key, "params": params, "contamination": contam, "model": model, "scores": sc,
                        "flags": fl, "metrics": met}
        st.session_state[f"{module.key}_cm_view"] = name
        S.log("Create", f"{name}: {met['n_flagged']} flagged")
        st.rerun()
    if not models:
        SH.empty(main, "No model created yet", "Choose a detector, set its parameters and the contamination, then "
                 "press Create model.")
        return
    SH.section(main, "Created models")
    rows = [{"Model": n, **{AL.get(k, k): v for k, v in r["metrics"].items()},
             "final": "✓" if S.u.get("chosen") == n else ""} for n, r in models.items()]
    main.dataframe(pd.DataFrame(rows).round(3), hide_index=True, width="stretch")
    names = list(models)
    vk = f"{module.key}_cm_view"
    if st.session_state.get(vk) not in names:
        st.session_state[vk] = names[-1]
    a, b = main.columns([3, 1], vertical_alignment="bottom")
    view = a.selectbox("Model", names, key=vk)
    is_final = S.u.get("chosen") == view
    if b.button("Final model ✓" if is_final else "Set as final model", key=f"{module.key}_setfinal",
                type="secondary" if is_final else "primary", use_container_width=True, disabled=is_final):
        for k in ("stab", "bundle", "meta", "deploy"):
            S.u.pop(k, None)
        S.u["chosen"] = view
        S.log("Final model", view)
        SH.advance(module, "analyze")
        st.rerun()
    r = models[view]
    fig_out(main, UP.score_hist(r["scores"], r["flags"], reference=ud.reference))


# =========================================================================== ANALYZE
def analyze_step(S, module, main, side):
    ud, Z = S.u["ud"], S.u["Z"]
    m = S.u["models"][S.u["chosen"]]
    P = SH.panel(side, "analyze")
    c = SH.card(P, f"{module.key}_plot", "Plot")
    all_ = SH.picker(c, f"{module.key}_an_all", "Show", ["One plot", "All plots"]) == "All plots"
    options = ["Score distribution", "Ranked scores", "Anomaly map (PCA)", "Anomaly map (t-SNE)",
                            "Top anomalies"]
    selected = options if all_ else [c.radio("Plot", options, key=f"{module.key}_plotsel", label_visibility="collapsed")]
    an = S.u.setdefault("analysis", {})
    main.markdown(f"**{S.u['chosen']}** · {int(m['flags'].sum())} of {len(Z):,} flagged")
    for plot in selected:
        if all_:
            SH.section(main, plot)
        if plot == "Score distribution":
            fig_out(main, UP.score_hist(m["scores"], m["flags"], reference=ud.reference))
        elif plot == "Ranked scores":
            fig_out(main, UP.ranked_scores(m["scores"], m["flags"]))
        elif plot.startswith("Anomaly map"):
            method = "tsne" if "t-SNE" in plot else "pca"
            if method not in an:
                with main.spinner("Computing the embedding…"):
                    an[method] = CL.embedding_2d(Z, method)
            E, ev = an[method]
            lab = np.where(m["flags"] == 1, "flagged", "not flagged")
            hover = ud.ids.astype(str).values if ud.ids is not None else None
            fig_out(main, UP.embedding(E, lab, "Flagged samples on the map", ev, hover, color_name="detector"))
        else:
            n = int(c.slider("Rows", 10, 200, 30, key=f"{module.key}_topn"))
            out = pd.DataFrame({"score": m["scores"], "flag": m["flags"]}, index=ud.X.index)
            if ud.ids is not None:
                out.insert(0, ud.ids.name, ud.ids.values)
            if ud.reference is not None:
                out["reference"] = ud.reference
            out = out.sort_values("score", ascending=False).head(n)
            main.dataframe(out.round(4), width="stretch", height=420)


# =========================================================================== VALIDATE
def validate_step(S, module, main, side):
    ud, Z = S.u["ud"], S.u["Z"]
    m = S.u["models"][S.u["chosen"]]
    P = SH.panel(side, "validate")
    c = SH.card(P, f"{module.key}_stab", "Flag stability", "Refits on random subsamples and records how often each "
                "sample is flagged.")
    nb = int(c.slider("Resamples", 20, 500, 100, 10, key=f"{module.key}_nb",
                      help="100 or more is recommended (Hennig's default for clusterwise stability is 100)."))
    frac = float(c.slider("Subsample fraction", 0.5, 0.95, 0.8, 0.05, key=f"{module.key}_frac"))
    if SH.run_button(P, module, "Run stability analysis", key_suffix="_stab"):
        with main.spinner(f"Refitting {nb} times…"):
            try:
                S.u["stab"] = (S.u["chosen"], AN.flag_stability(Z, m["key"], m["contamination"], m["params"], nb, frac))
            except Exception as e:  # noqa: BLE001
                main.error(f"Stability analysis failed: {e}"); return
        st.rerun()
    met = m["metrics"]
    if ud.reference is not None:
        SH.section(main, "Agreement with the reference label")
        SH.kpis(main, [(AL[k], f"{met[k]:.3f}") for k in REF_METRICS if k in met])
        main.caption("Valid only if the reference truly marks anomalies of the kind you care about. Precision@n uses "
                     "n = number of reference anomalies.")
    st_ = S.u.get("stab")
    if not st_ or st_[0] != S.u["chosen"]:
        SH.empty(main, "Stability not assessed yet", "Without labels, the best available check is whether the same "
                 "samples are flagged when the detector is refitted on different subsets of the data.")
        return
    r = st_[1]
    SH.section(main, "Flag stability")
    SH.kpis(main, [("flag overlap (Jaccard)", f"{r['mean_jaccard']:.2f} ± {r['sd_jaccard']:.2f}"),
                   ("stable flags (≥ 80% of refits)", f"{r['stable_flags']} of {r['n_flagged']}"),
                   ("resamples", str(r["n_boot"]))])
    freq = r["flag_frequency"]
    import plotly.graph_objects as go
    fig = go.Figure(go.Histogram(x=freq[freq > 0], nbinsx=20, marker_color=module.color))
    fig.update_layout(title="How often each ever-flagged sample is flagged across refits", xaxis_title="flag frequency",
                      xaxis_tickformat=".0%", yaxis_title="samples", height=340)
    fig_out(main, fig)
    if AN.get(m["key"]).transductive:
        main.caption("Batch-only detector: each sample's frequency is computed over the subsamples that contained it.")


# =========================================================================== EXPLAIN
def explain_step(S, module, main, side):
    ud = S.u["ud"]
    m = S.u["models"][S.u["chosen"]]
    P = SH.panel(side, "explain")
    order = np.argsort(-m["scores"])
    flagged = [i for i in order if m["flags"][i] == 1] or list(order[:20])
    lab = (lambda i: f"{ud.ids.iloc[i]} (score {m['scores'][i]:.3g})") if ud.ids is not None else \
        (lambda i: f"row {ud.X.index[i]} (score {m['scores'][i]:.3g})")
    c = SH.card(P, f"{module.key}_xs", "Sample", "Flagged samples, most anomalous first.")
    i = c.selectbox("Sample", flagged, format_func=lab, key=f"{module.key}_xsample", label_visibility="collapsed")
    n = int(c.slider("Features shown", 5, 40, 15, key=f"{module.key}_xn"))
    if not ud.numeric:
        main.info("Deviation profiles need numeric features."); return
    prof = AN.deviation_profile(ud.X, ud.numeric, ud.X.index[i])
    fig_out(main, UP.deviation_bars(prof, n, f"Why is {lab(i).split(' (')[0]} unusual?"))
    main.caption("Robust z = (value − median) / (1.4826 × MAD) across all samples. It shows which single features are "
                 "extreme for this sample; detectors can also flag unusual combinations that no single feature shows.")
    main.dataframe(prof.head(n).round(3), hide_index=True, width="stretch")


# =========================================================================== SCORE & DEPLOY
def deploy_step(S, module, main, side):
    ud, Z = S.u["ud"], S.u["Z"]
    name = S.u["chosen"]
    m = S.u["models"][name]
    spec = AN.get(m["key"])
    P = SH.panel(side, "deploy")
    c = SH.card(P, f"{module.key}_fin", "Bundle", "Preprocessing and the fitted detector, signed by this server.")
    if spec.transductive:
        c.warning(f"{spec.name} cannot score new samples. Choose an inductive detector to deploy.")
    if SH.run_button(P, module, "Create bundle", disabled=spec.transductive, key_suffix="_finalize"):
        obj = {"pre": S.u["pre"], "model": m["model"], "key": m["key"]}
        meta = {"module": "anomaly", "task": "anomaly", "model": name, "model_key": m["key"],
                "contamination": m["contamination"], "n_train": len(Z), "features": list(ud.X.columns),
                "numeric": ud.numeric, "categorical": ud.categorical, "sample_id": S.u["roles"]["sample_id"],
                "profile": MM.training_profile(ud.X, ud.numeric, ud.categorical),
                "fingerprint": MM.dataset_fingerprint(ud.X), "metrics": {k: float(v) for k, v in m["metrics"].items()}}
        S.u["bundle"] = MM.make_bundle(obj, meta)
        S.u["meta"], S.u["deploy"] = MM.read_metadata(S.u["bundle"]), obj
        S.log("Bundle", name)
        st.rerun()
    c = SH.card(P, f"{module.key}_load", "Or load a bundle")
    up = c.file_uploader("Bundle", type=["omnix", "zip"], key=f"{module.key}_bundleup", label_visibility="collapsed")
    if up is not None and c.button("Load bundle", key=f"{module.key}_loadb", use_container_width=True):
        try:
            obj, _, meta = MM.load_bundle(up.getvalue())
            if meta.get("module") != "anomaly":
                raise MM.BundleError("This bundle is not an anomaly-detection model.")
            S.u["deploy"], S.u["meta"], S.u["bundle"] = obj, meta, up.getvalue()
            st.rerun()
        except MM.BundleError as e:
            c.error(str(e))
    SH.section(main, "Scores for the fitted samples")
    out = pd.DataFrame({"score": m["scores"], "flag": m["flags"]}, index=ud.X.index)
    if ud.ids is not None:
        out.insert(0, ud.ids.name, ud.ids.values)
    main.dataframe(out.sort_values("score", ascending=False).head(200).round(4), width="stretch", height=260)
    downloads(main, out.reset_index(drop=True), "anomaly_scores", f"{module.key}_scdl")
    meta = S.u.get("meta")
    if meta is None:
        SH.empty(main, "No bundle yet", "Create a bundle to download the detector and score new samples against "
                 "the training data.")
        return
    SH.section(main, "Model bundle")
    SH.kpis(main, [("model", meta["model"]), ("trained on", f"{meta['n_train']:,} samples"), ("created", meta["created"])])
    if S.u.get("bundle"):
        main.download_button("Download model bundle", S.u["bundle"], f"anomaly_{meta['fingerprint'][:8]}.omnix",
                             "application/zip", key=f"{module.key}_bundledl", type="primary")
    SH.section(main, "Score new samples")
    new = main.file_uploader("New data", type=["csv", "tsv", "txt", "xlsx"], key=f"{module.key}_newup")
    if new is None:
        return
    try:
        nd = load_table(new.getvalue(), new.name)
        obj = S.u["deploy"]
        Zn = pd.DataFrame(np.asarray(obj["pre"].transform(MM.prepare_features(nd, meta)), float))
        sc, fl = AN.score_new(obj["model"], Zn, obj["key"])
    except (DataLoadError, AN.TransductiveError, Exception) as e:  # noqa: BLE001
        main.error(f"Scoring failed: {e}"); return
    res = pd.DataFrame({"score": sc, "flag": fl}, index=nd.index)
    sid = meta.get("sample_id")
    if sid and sid in nd:
        res.insert(0, sid, nd[sid].values)
    main.caption(f"Threshold fixed at training time ({meta['contamination']:.0%} contamination of the training data). "
                 f"{int(fl.sum())} of {len(fl)} new samples flagged.")
    main.dataframe(res.round(4), width="stretch", height=300)
    downloads(main, res.reset_index(drop=True), "new_anomaly_scores", f"{module.key}_newdl")
