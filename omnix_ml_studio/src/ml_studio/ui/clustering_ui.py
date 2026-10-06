"""Clustering: Compare, Create & tune, Analyze, Validate, Explain, Assign & deploy."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from ..core import clustering as CL
from ..core import model_manager as MM
from ..core.data_validation import DataLoadError, load_table
from ..visualization import unsup_plots as UP
from . import jobs
from . import shell as SH
from .components import downloads, model_picker, fig_out
from .unsup_common import clear, spec_param_editor

FAMILIES = ["Centroid", "Model-based", "Hierarchical", "Graph", "Density", "Exemplar"]
INTERNAL = ["silhouette", "calinski_harabasz", "davies_bouldin"]
EXTERNAL = ["ari", "ami", "homogeneity", "completeness", "v_measure"]
LOWER = {"davies_bouldin"}
L = CL.CLUSTER_METRIC_LABELS


def _metric_table(lb, mets):
    tab = pd.DataFrame({"Model": lb["Model"].values})
    for m in ["n_clusters", "noise_fraction"] + mets:
        if m in lb:
            tab[L.get(m, m)] = pd.to_numeric(lb[m], errors="coerce").values
    tab["Time (s)"] = lb["fit_time_s"].values
    return tab


def _styled(tab, mets, module):
    fmt = {L[m]: "{:.3f}" for m in mets if L[m] in tab}
    fmt.update({"Noise fraction": "{:.1%}", "Time (s)": "{:.2f}", "Calinski–Harabasz": "{:.1f}"})
    sty = tab.style.format(fmt, na_rep="—")
    for m in mets:
        if L[m] in tab:
            fn = sty.highlight_min if m in LOWER else sty.highlight_max
            sty = fn(subset=[L[m]], props=f"background-color:{module.tint}; font-weight:600; color:{module.color}")
    return sty


# =========================================================================== COMPARE
def compare_step(S, module, main, side):
    ud, Z = S.u["ud"], S.u["Z"]
    P = SH.panel(side, "compare")
    specs = dict(CL.REGISTRY)
    c = SH.card(P, f"{module.key}_algos", "Algorithms", f"{len(specs)} available.")
    keys = model_picker(c, f"{module.key}_pick", specs, FAMILIES, CL.DEFAULT,
                        badge=lambda s: "" if s.uses_k else "finds k itself", columns=1)
    c = SH.card(P, f"{module.key}_k", "Number of clusters", "Used by algorithms that need k. Choose it properly in "
                "Create & tune with a k scan.")
    k = int(c.number_input("k", 2, 30, 4, key=f"{module.key}_cmpk"))
    if len(Z) > 5000:
        c.warning("Spectral, OPTICS, Mean Shift and Affinity Propagation scale poorly beyond a few thousand samples.")
    jkey = f"{module.key}_compare"

    def apply(job):
        lb, fail, kk = job.result
        clear(S, "compare")
        S.u["lb"], S.u["fail"], S.u["cmp_k"] = lb, fail, kk
        S.log("Compare", f"{len(lb)} algorithms, k={kk}")
    job = jobs.collect(jkey, apply, id(Z))
    busy = job is not None and job.running
    if SH.run_button(P, module, "Running in the background…" if busy else f"Compare {len(keys)} algorithms",
                     disabled=not keys or busy, key_suffix="_compare"):
        def work(job, keys=tuple(keys), k=k):
            r = jobs.run_stoppable(job, CL.compare_clustering, Z, list(keys), k, ud.reference, 0,
                                   progress=jobs.stoppable_progress(job))
            return None if r is None else (*r, k)
        jobs.start(jkey, f"Comparing {len(keys)} algorithms", work, signature=id(Z))
        st.rerun()
    if job is not None and job.status == "error":
        main.error(f"The comparison did not finish: {job.error}")
    with main:
        jobs.monitor(jkey)
    if busy:
        return
    SH.kpis(main, [("samples", f"{len(Z):,}"), ("model dimensions", str(Z.shape[1])),
                   ("reference label", ud.reference_name or "none")])
    lb = S.u.get("lb")
    if lb is None:
        SH.empty(main, "No comparison yet", "Choose algorithms and press Compare. Each is fitted to the same "
                 "preprocessed data and scored with internal criteria (and the reference label, if given).")
        return
    if lb.empty:
        main.error("Every algorithm failed: " + "; ".join(list(S.u["fail"].values())[:3]))
        return
    mets = INTERNAL + ([m for m in EXTERNAL if m in lb] if ud.reference is not None else [])
    SH.section(main, "Leaderboard", f"k = {S.u['cmp_k']} for algorithms that need it. Silhouette and Calinski–"
                                    "Harabasz: higher is better; Davies–Bouldin: lower is better. Internal criteria "
                                    "favour compact, convex clusters and often disagree with each other.")
    main.dataframe(_styled(_metric_table(lb, mets), mets, module), hide_index=True, width="stretch")
    if ud.reference is not None:
        main.caption("ARI, AMI, homogeneity, completeness and V-measure compare clusters with the reference label. "
                     "Agreement with a known label is a sanity check, not proof that the clusters are real.")
    if S.u["fail"]:
        with main.expander(f"{len(S.u['fail'])} algorithm(s) failed"):
            for kk, msg in S.u["fail"].items():
                st.markdown(f"**{CL.get(kk).name}**: {msg}")
    downloads(main, lb, "clustering_comparison", f"{module.key}_lbdl")
    best = lb.iloc[0]
    if main.button(f"Create and tune {best['Model']}", key=f"{module.key}_tobest", type="primary"):
        st.session_state[f"{module.key}_cm_key"] = best["key"]
        SH.go_to(module, "create")
        st.rerun()


# =========================================================================== CREATE & TUNE
def create_step(S, module, main, side):
    ud, Z = S.u["ud"], S.u["Z"]
    models = S.u.setdefault("models", {})
    P = SH.panel(side, "create")
    keys = list(CL.REGISTRY)
    if S.u.get("lb") is not None and len(S.u["lb"]):
        keys = list(S.u["lb"]["key"]) + [k for k in keys if k not in set(S.u["lb"]["key"])]
    ck = f"{module.key}_cm_key"
    if st.session_state.get(ck) not in keys:
        st.session_state[ck] = keys[0]
    c = SH.card(P, f"{module.key}_cm", "Algorithm")
    key = c.selectbox("Algorithm", keys, format_func=lambda k: CL.get(k).name, key=ck, label_visibility="collapsed")
    spec = CL.get(key)
    if spec.notes:
        c.caption(spec.notes)
    if spec.uses_k:
        c2 = SH.card(P, f"{module.key}_scan", "Choose k", "Fits k = 2 … max and plots every criterion.")
        kmax = int(c2.slider("Largest k", 4, 20, 10, key=f"{module.key}_kmax"))
        if c2.button("Run k scan", key=f"{module.key}_runscan", use_container_width=True):
            with main.spinner("Scanning k…"):
                try:
                    S.u["scan"] = (key, CL.k_scan(Z, key, range(2, kmax + 1)))
                except Exception as e:  # noqa: BLE001
                    main.error(f"k scan failed: {e}")
            st.rerun()
    c = SH.card(P, f"{module.key}_hp", "Parameters")
    params = spec_param_editor(c, f"{module.key}_pe_{key}", spec.params)
    if SH.run_button(P, module, "Create model", key_suffix="_create"):
        try:
            with main.spinner(f"Fitting {spec.name}…"):
                model, lab = CL.fit_predict(Z, key, params)
        except Exception as e:  # noqa: BLE001
            main.error(f"{spec.name} failed: {type(e).__name__}: {e}"); return
        desc = ", ".join(f"{k}={v}" for k, v in params.items()) or "defaults"
        name = f"{spec.name} ({desc})"
        m = CL.internal_metrics(Z, lab)
        if ud.reference is not None:
            m.update(CL.external_metrics(lab, ud.reference))
        models[name] = {"key": key, "params": params, "labels": lab, "model": model, "metrics": m}
        st.session_state[f"{module.key}_cm_view"] = name
        S.log("Create", f"{name}: {m['n_clusters']} clusters")
        st.rerun()

    scan = S.u.get("scan")
    if scan and scan[0] == key:
        SH.section(main, "k scan", "Criteria often disagree, especially with noise features or unequal cluster sizes. "
                                   "Prefer a k that is supported by several criteria and by stability (Validate).")
        fig_out(main, UP.k_scan_plot(scan[1]))
    if not models:
        if not (scan and scan[0] == key):
            SH.empty(main, "No model created yet", "Set the parameters in the panel and press Create model. For "
                     "algorithms that need k, run a k scan first.")
        return
    SH.section(main, "Created models")
    rows = []
    for n, r in models.items():
        rows.append({"Model": n, **{L.get(k, k): v for k, v in r["metrics"].items() if k in L},
                     "final": "✓" if S.u.get("chosen") == n else ""})
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
        for k in ("stab", "analysis", "bundle", "meta", "deploy"):
            S.u.pop(k, None)
        S.u["chosen"] = view
        S.log("Final model", view)
        SH.advance(module, "analyze")
        st.rerun()
    fig_out(main, UP.cluster_sizes(models[view]["labels"]))


# =========================================================================== ANALYZE
def analyze_step(S, module, main, side):
    ud, Z = S.u["ud"], S.u["Z"]
    m = S.u["models"][S.u["chosen"]]
    lab = m["labels"]
    P = SH.panel(side, "analyze")
    c = SH.card(P, f"{module.key}_plot", "Plot")
    plots = ["Cluster map (PCA)", "Cluster map (t-SNE)", "Cluster sizes", "Silhouette"] + \
        (["Versus reference"] if ud.reference is not None else [])
    all_ = SH.picker(c, f"{module.key}_an_all", "Show", ["One plot", "All plots"]) == "All plots"
    options = plots
    selected = options if all_ else [c.radio("Plot", options, key=f"{module.key}_plotsel", label_visibility="collapsed")]
    an = S.u.setdefault("analysis", {})
    main.markdown(f"**{S.u['chosen']}** · {m['metrics']['n_clusters']} clusters")
    for plot in selected:
        if all_:
            SH.section(main, plot)
        hover = ud.ids.astype(str).values if ud.ids is not None else None
        if plot.startswith("Cluster map"):
            method = "tsne" if "t-SNE" in plot else "pca"
            if method not in an:
                with main.spinner("Computing the embedding…"):
                    an[method] = CL.embedding_2d(Z, method)
            E, ev = an[method]
            col = SH.picker(main, f"{module.key}_mapcol_{method}", "Colour by", ["Cluster"] + (["Reference"] if ud.reference is not None else []))
            if col == "Cluster":
                fig_out(main, UP.embedding(E, lab, f"Clusters on the {'PCA' if method == 'pca' else 't-SNE'} map", ev, hover))
            else:
                fig_out(main, UP.embedding(E, pd.Series(ud.reference).astype(str).values, "Reference label", ev, hover,
                                               color_name=ud.reference_name))
            if method == "tsne":
                main.caption("t-SNE preserves local neighbourhoods; distances between far-apart groups and group sizes on "
                             "the map are not meaningful.")
        elif plot == "Cluster sizes":
            fig_out(main, UP.cluster_sizes(lab))
        elif plot == "Silhouette":
            if m["metrics"]["n_clusters"] < 2:
                main.info("Silhouette needs at least two clusters.")
            else:
                fig_out(main, UP.silhouette_plot(Z, lab))
                main.caption("Negative values: samples closer to another cluster than to their own.")
        else:
            fig_out(main, UP.contingency(lab, ud.reference))


# =========================================================================== VALIDATE
def validate_step(S, module, main, side):
    Z = S.u["Z"]
    m = S.u["models"][S.u["chosen"]]
    P = SH.panel(side, "validate")
    c = SH.card(P, f"{module.key}_stab", "Bootstrap stability", "Refits on random subsamples and checks whether each "
                "cluster is recovered (Hennig's clusterwise Jaccard).")
    nb = int(c.slider("Resamples", 20, 500, 100, 10, key=f"{module.key}_nb",
                      help="100 or more is recommended (Hennig's default for clusterwise stability is 100)."))
    frac = float(c.slider("Subsample fraction", 0.5, 0.95, 0.8, 0.05, key=f"{module.key}_frac"))
    if SH.run_button(P, module, "Run stability analysis", key_suffix="_stab"):
        with main.spinner(f"Refitting {nb} times…"):
            try:
                S.u["stab"] = (S.u["chosen"], CL.bootstrap_stability(Z, m["key"], m["params"], nb, frac))
            except Exception as e:  # noqa: BLE001
                main.error(f"Stability analysis failed: {e}"); return
        S.log("Stability", f"{nb} resamples")
        st.rerun()
    mm = m["metrics"]
    SH.kpis(main, [("clusters", str(mm["n_clusters"])), ("silhouette", f"{mm.get('silhouette', np.nan):.3f}"),
                   ("noise", f"{mm['noise_fraction']:.1%}")] +
            ([("ARI vs reference", f"{mm['ari']:.3f}")] if "ari" in mm else []))
    st_ = S.u.get("stab")
    if not st_ or st_[0] != S.u["chosen"]:
        SH.empty(main, "Stability not assessed yet", "Clustering algorithms always return clusters, even in data "
                 "with no group structure. Stability under resampling is the main evidence that a cluster is "
                 "a property of the data rather than of the algorithm.")
        return
    r = st_[1]
    fig_out(main, UP.jaccard_bars(r["per_cluster"]))
    main.markdown(f"Agreement between full-data and subsample clusterings: ARI **{r['ari_mean']:.3f} ± {r['ari_sd']:.3f}** "
                  f"({r['n_boot']} subsamples of {r['fraction']:.0%}).")
    main.dataframe(r["per_cluster"].round(3), hide_index=True, width="stretch")
    main.caption("Jaccard ≥ 0.75 stable, 0.6–0.75 a pattern, 0.5–0.6 weak, < 0.5 dissolved (Hennig 2007). Stability "
                 "is necessary but not sufficient: a stable cluster can still reflect a batch effect or artefact.")


# =========================================================================== EXPLAIN
def explain_step(S, module, main, side):
    ud = S.u["ud"]
    m = S.u["models"][S.u["chosen"]]
    P = SH.panel(side, "explain")
    c = SH.card(P, f"{module.key}_prof", "Profiles")
    n = int(c.slider("Features in the heatmap", 5, 60, 20, key=f"{module.key}_pn"))
    an = S.u.setdefault("analysis", {})
    if an.get("prof_for") != S.u["chosen"]:
        with main.spinner("Comparing features across clusters…"):
            an["prof"] = CL.cluster_profiles(ud.X, m["labels"], ud.numeric, ud.categorical)
            an["prof_for"] = S.u["chosen"]
    prof = an["prof"]
    main.warning("Descriptive only. The clusters were built from these same features, so differences between them "
                 "are expected by construction; the p- and q-values are not valid tests of a hypothesis.")
    top = [f for f in prof.loc[prof["type"] == "numeric", "feature"].head(n)]
    if top:
        fig_out(main, UP.profile_heatmap(ud.X, m["labels"], top))
    main.dataframe(prof.round(4), hide_index=True, width="stretch", height=360)
    downloads(main, prof, "cluster_profiles", f"{module.key}_profdl")


# =========================================================================== ASSIGN & DEPLOY
def deploy_step(S, module, main, side):
    ud, Z = S.u["ud"], S.u["Z"]
    name = S.u["chosen"]
    m = S.u["models"][name]
    spec = CL.get(m["key"])
    P = SH.panel(side, "deploy")
    c = SH.card(P, f"{module.key}_fin", "Bundle", "Preprocessing, the fitted model and its training assignment, "
                "signed by this server.")
    if SH.run_button(P, module, "Create bundle", key_suffix="_finalize"):
        obj = {"pre": S.u["pre"], "model": m["model"], "key": m["key"], "Z": Z, "labels": m["labels"]}
        meta = {"module": "clustering", "task": "clustering", "model": name, "model_key": m["key"],
                "params": {k: str(v) for k, v in m["params"].items()}, "n_train": len(Z),
                "features": list(ud.X.columns), "numeric": ud.numeric, "categorical": ud.categorical,
                "sample_id": S.u["roles"]["sample_id"],
                "profile": MM.training_profile(ud.X, ud.numeric, ud.categorical),
                "fingerprint": MM.dataset_fingerprint(ud.X), "metrics": {k: float(v) for k, v in m["metrics"].items()},
                "assignment": "native predict" if spec.can_predict else "nearest training sample"}
        S.u["bundle"] = MM.make_bundle(obj, meta)
        S.u["meta"], S.u["deploy"] = MM.read_metadata(S.u["bundle"]), obj
        S.log("Bundle", name)
        st.rerun()
    c = SH.card(P, f"{module.key}_load", "Or load a bundle")
    up = c.file_uploader("Bundle", type=["omnix", "zip"], key=f"{module.key}_bundleup", label_visibility="collapsed")
    if up is not None and c.button("Load bundle", key=f"{module.key}_loadb", use_container_width=True):
        try:
            obj, _, meta = MM.load_bundle(up.getvalue())
            if meta.get("module") != "clustering":
                raise MM.BundleError("This bundle is not a clustering model.")
            S.u["deploy"], S.u["meta"], S.u["bundle"] = obj, meta, up.getvalue()
            st.rerun()
        except MM.BundleError as e:
            c.error(str(e))

    SH.section(main, "Cluster assignments", "Labels for the samples used to fit the model.")
    out = pd.DataFrame({"cluster": m["labels"]}, index=ud.X.index)
    if ud.ids is not None:
        out.insert(0, ud.ids.name, ud.ids.values)
    main.dataframe(out.head(200), width="stretch", height=260)
    downloads(main, out.reset_index(drop=True), "cluster_assignments", f"{module.key}_asdl")
    meta = S.u.get("meta")
    if meta is None:
        SH.empty(main, "No bundle yet", "Create a bundle to download the model and assign new samples to these "
                 "clusters.")
        return
    SH.section(main, "Model bundle")
    SH.kpis(main, [("model", meta["model"]), ("assignment", meta["assignment"]), ("created", meta["created"])])
    if S.u.get("bundle"):
        main.download_button("Download model bundle", S.u["bundle"], f"clustering_{meta['fingerprint'][:8]}.omnix",
                             "application/zip", key=f"{module.key}_bundledl", type="primary")
    SH.section(main, "Assign new samples")
    new = main.file_uploader("New data", type=["csv", "tsv", "txt", "xlsx"], key=f"{module.key}_newup")
    if new is None:
        return
    try:
        nd = load_table(new.getvalue(), new.name)
        obj = S.u["deploy"]
        Xn = MM.prepare_features(nd, meta)
        Zn = pd.DataFrame(np.asarray(obj["pre"].transform(Xn), float))
        lab, how = CL.assign_new(obj["model"], obj["key"], obj["Z"], obj["labels"], Zn)
    except (DataLoadError, Exception) as e:  # noqa: BLE001
        main.error(f"Assignment failed: {type(e).__name__}: {e}"); return
    chk = MM.check_new_data(nd, meta["profile"])
    res = pd.DataFrame({"cluster": lab, "out_of_range_fraction": chk["frac_out_of_range"].values}, index=nd.index)
    sid = meta.get("sample_id")
    if sid and sid in nd:
        res.insert(0, sid, nd[sid].values)
    if chk["missing"]:
        main.warning(f"{len(chk['missing'])} feature(s) missing and imputed: {chk['missing'][:10]}")
    main.caption(f"Assigned by {how}. New samples never change the clusters.")
    main.dataframe(res, width="stretch", height=300)
    downloads(main, res.reset_index(drop=True), "new_cluster_assignments", f"{module.key}_newdl")
