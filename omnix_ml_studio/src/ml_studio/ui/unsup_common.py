"""Shared pieces for the unsupervised modules (clustering, anomaly detection) and time series."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import streamlit as st

from ..core.clustering import build_unsupervised_preprocessor, prepare_unsupervised
from ..core.config import PreprocessConfig
from ..core.data_validation import MAX_UPLOAD_MB, DataLoadError, infer_types, load_table
from ..core.preprocessing import check_log_validity, describe_preprocessing
from ..visualization import feature_plots as FP
from . import shell as SH
from .components import fig_out

EXAMPLES = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "examples"))
NONE = "(none)"
EXAMPLE_FOR = {"clustering": ["clustering.csv", "clustering_well_separated.csv", "clinical_plus_800_genes.csv",
                              "multiclass_classification.csv"],
               "anomaly": ["anomaly.csv", "anomaly_subtle.csv", "binary_classification.csv", "clinical_plus_800_genes.csv"],
               "timeseries": ["monthly_series.csv"]}
REF_HINT = {"clustering": ("true_subtype", "subtype", "label", "class", "cluster"),
            "anomaly": ("is_anomaly", "anomaly", "outlier", "label")}
# results cleared when an upstream stage changes
U_STAGES = {"data": ["ud", "roles", "qual", "series"], "setup": ["Z", "pre", "cfg", "setup", "design"],
            "compare": ["lb", "fail", "cmp", "cvs"], "model": ["models", "chosen", "scan", "stab", "analysis",
                                                               "bundle", "meta", "deploy", "holdout", "final"]}


def clear(S, stage: str):
    order = list(U_STAGES)
    for s in order[order.index(stage):]:
        for k in U_STAGES[s]:
            S.u.pop(k, None)


def spec_param_editor(parent, key: str, params: dict) -> dict:
    """Editor for registry params given as name -> (default, kind, lo, hi) or (default, 'choice', [options])."""
    out = {}
    for name, spec in params.items():
        default, kind = spec[0], spec[1]
        wk = f"{key}_{name}"
        if kind == "choice":
            opts = spec[2]
            out[name] = parent.selectbox(name, opts, index=opts.index(default) if default in opts else 0, key=wk)
        elif kind == "int":
            lo, hi = spec[2], spec[3]
            if default is None:
                on = parent.checkbox(f"Set {name}", key=f"{wk}_on")
                if on:
                    out[name] = int(parent.number_input(name, lo, hi, min(max(lo, 12), hi), key=wk))
            else:
                out[name] = int(parent.number_input(name, lo, hi, int(default), key=wk))
        elif kind == "float":
            lo, hi = float(spec[2]), float(spec[3])
            if default is None:
                on = parent.checkbox(f"Set {name}", key=f"{wk}_on", help="Estimated from the data when unset.")
                if on:
                    out[name] = float(parent.number_input(name, lo, hi, (lo + hi) / 4, format="%.4g", key=wk))
            else:
                out[name] = float(parent.number_input(name, lo, hi, float(default), format="%.4g", key=wk))
    return out


# =========================================================================== DATA
def source_card(S, module, P) -> bool:
    c = SH.card(P, f"{module.key}_src", "Dataset", f"CSV, TSV or Excel up to {MAX_UPLOAD_MB} MB.")
    src = c.radio("Source", ["Upload a file", "Example dataset"], horizontal=True, key=f"{module.key}_src",
                  label_visibility="collapsed")
    if src == "Upload a file":
        up = c.file_uploader("File", type=["csv", "tsv", "txt", "xlsx"], key=f"{module.key}_upload",
                             label_visibility="collapsed")
        if up is not None and up.name != S.filename:
            try:
                S.df, S.filename, S.source = load_table(up.getvalue(), up.name), up.name, "upload"
                S.invalidate("design")
            except (DataLoadError, Exception) as e:  # noqa: BLE001
                c.error(f"Could not read the file: {e}")
    else:
        files = [f for f in EXAMPLE_FOR[module.key] if os.path.exists(os.path.join(EXAMPLES, f))]
        pick = c.selectbox("Example", files, key=f"{module.key}_example")
        if c.button("Load example", key=f"{module.key}_load_ex", use_container_width=True):
            S.df, S.filename, S.source = load_table(os.path.join(EXAMPLES, pick)), pick, "example"
            S.invalidate("design")
            st.rerun()
    return S.df is not None


def data_step(S, module, main, side):
    P = SH.panel(side, "data")
    if not source_card(S, module, P):
        SH.empty(main, "No dataset yet", "Upload a file or load an example in the panel. Each row is one sample and "
                 "each column a feature. No outcome is needed; an optional reference label can be kept aside for "
                 "evaluation only.")
        return
    df = S.df
    cols = list(df.columns)
    c = SH.card(P, f"{module.key}_roles", "Column roles", "Features are used for fitting. The reference label is "
                "never used for fitting, only to evaluate the result.")
    opt = [NONE] + cols
    lower = {x.lower(): x for x in cols}
    sid_guess = next((lower[k] for k in ("sample_id", "id", "patient_id", "sample") if k in lower), NONE)
    ref_guess = next((lower[k] for k in REF_HINT[module.key] if k in lower), NONE)
    a, b = c.columns(2)
    sid = a.selectbox("Sample ID", opt, index=opt.index(sid_guess), key=f"{module.key}_sid")
    ref = b.selectbox("Reference label", opt, index=opt.index(ref_guess), key=f"{module.key}_ref",
                      help="Known groups or anomaly flags, if any. Kept out of the model.")
    cand = [x for x in cols if x not in {sid, ref}]
    mode = c.radio("Features", ["All other columns", "Exclude some", "Choose"], horizontal=True,
                   key=f"{module.key}_fmode")
    if mode == "All other columns":
        feats = cand
    elif mode == "Exclude some":
        ex = c.multiselect("Exclude", cand, key=f"{module.key}_fex")
        feats = [x for x in cand if x not in ex]
    else:
        feats = c.multiselect("Features", cand, default=cand[: min(20, len(cand))], key=f"{module.key}_fpick")
    num, cat = infer_types(df, feats)
    cat_over = []
    with c.expander("Types"):
        cat_over = st.multiselect("Treat as categorical", num, key=f"{module.key}_catover")
    c.caption(f"{len(feats)} features: {len(num) - len(cat_over)} numeric, {len(cat) + len(cat_over)} categorical")
    if module.key == "anomaly" and ref != NONE:
        r = df[ref]
        if not set(pd.Series(r.dropna().unique()).astype(str)) <= {"0", "1", "True", "False", "0.0", "1.0"}:
            c.warning("The anomaly reference should be 0/1 (1 = anomaly).")
    if SH.run_button(P, module, "Confirm roles", disabled=not feats, key_suffix="_roles"):
        clear(S, "data")
        ud = prepare_unsupervised(df, feats, None if ref == NONE else ref, None if sid == NONE else sid, cat_over)
        if ud.reference is not None and module.key == "anomaly":
            ud.reference = pd.Series(ud.reference).astype(str).isin(["1", "True", "1.0"]).astype(int).to_numpy()
        S.u["ud"], S.u["roles"] = ud, {"sample_id": None if sid == NONE else sid, "reference": None if ref == NONE else ref,
                                       "features": feats}
        S.log("Roles", f"{len(feats)} features" + (f"; reference {ref}" if ref != NONE else ""))
        SH.advance(module, "setup")
        st.rerun()

    SH.kpis(main, [("rows", f"{df.shape[0]:,}"), ("columns", f"{df.shape[1]:,}"),
                   ("missing cells", f"{df.isna().values.mean():.1%}"), ("file", S.filename)])
    with main.expander("Preview", expanded="ud" not in S.u):
        ncol = min(df.shape[1], 40)
        st.dataframe(df.iloc[:30, :ncol], width="stretch", height=260)
    ud = S.u.get("ud")
    if ud is None:
        main.info("Assign the column roles in the panel and confirm them.")
        return
    X = ud.X
    SH.section(main, "Data quality", "Problems that matter for distance- and density-based methods.")
    miss = X.isna().mean()
    const = [x for x in ud.numeric if X[x].nunique(dropna=True) <= 1]
    scale = X[ud.numeric].std().replace(0, np.nan) if ud.numeric else pd.Series(dtype=float)
    ratio = float(scale.max() / scale.min()) if len(scale.dropna()) > 1 else 1.0
    SH.kpis(main, [("samples", f"{len(X):,}"), ("features", f"{X.shape[1]:,}"),
                   ("with missing values", str(int((miss > 0).sum()))), ("constant", str(len(const)))])
    if ratio > 20:
        main.warning(f"Feature scales differ by a factor of {ratio:,.0f}. Distance-based methods will be dominated by "
                     "the largest-scale features unless you scale them in Setup.")
    if const:
        main.warning(f"{len(const)} constant feature(s) carry no information: {const[:8]}")
    if ud.reference is not None:
        vc = pd.Series(ud.reference).value_counts()
        main.caption(f"Reference '{ud.reference_name}': " + ", ".join(f"{k}: {v}" for k, v in vc.head(8).items()))
    v = SH.picker(main, f"{module.key}_dview", "View", ["Distributions", "Missing values", "Correlation"])
    if v == "Distributions" and ud.numeric:
        fig_out(main, FP.variance_summary(X[ud.numeric]))
    elif v == "Missing values":
        if X.isna().values.any():
            fig_out(main, FP.missingness_heatmap(X))
        else:
            main.success("No missing values.")
    elif v == "Correlation" and len(ud.numeric) >= 2:
        fig_out(main, FP.correlation_heatmap(X[ud.numeric], min(60, len(ud.numeric))))


# =========================================================================== SETUP
def setup_step(S, module, main, side):
    ud = S.u["ud"]
    P = SH.panel(side, "setup")
    p0 = S.u.get("cfg", PreprocessConfig(scaling="standard"))
    c = SH.card(P, f"{module.key}_imp", "Missing values")
    IMP = ["median", "mean", "knn", "constant"]
    imp = c.selectbox("Numeric", IMP, index=IMP.index(p0.numeric_imputation) if p0.numeric_imputation in IMP else 0,
                      key=f"{module.key}_imp")
    cimp = c.selectbox("Categorical", ["most_frequent", "constant"], key=f"{module.key}_cimp",
                       format_func=lambda x: "most frequent" if x == "most_frequent" else "'missing' category")
    c = SH.card(P, f"{module.key}_tr", "Transformation and scaling",
                "Scaling matters: most methods here measure distances.")
    TR = ["none", "log2", "yeo-johnson", "quantile"]
    tr = c.selectbox("Transformation", TR, index=TR.index(p0.transform) if p0.transform in TR else 0, key=f"{module.key}_tr")
    offset, problem = None, None
    if tr == "log2" and ud.numeric:
        chk = check_log_validity(ud.X, ud.numeric)
        if not chk["valid_without_offset"]:
            offset = c.number_input("Offset c in log2(x + c)", value=1.0, min_value=0.0, key=f"{module.key}_off")
            if offset + float(np.nanmin(ud.X[ud.numeric].to_numpy(float))) <= 0:
                problem = "the offset leaves non-positive values; choose a larger offset or Yeo-Johnson."
    SC = ["standard", "robust", "minmax", "none"]
    sc = c.selectbox("Scaling", SC, index=SC.index(p0.scaling), key=f"{module.key}_sc")
    c = SH.card(P, f"{module.key}_flt", "Feature filtering")
    mm = c.slider("Drop features missing in more than", 0.05, 1.0, p0.max_missing_fraction or 1.0, 0.05,
                  key=f"{module.key}_mm")
    vt = c.checkbox("Drop constant features", value=True, key=f"{module.key}_vt")
    cf_on = c.checkbox("Correlation filter", value=p0.correlation_threshold is not None, key=f"{module.key}_cfon")
    cf = c.slider("|r| above", 0.7, 0.99, p0.correlation_threshold or 0.95, 0.01, disabled=not cf_on, key=f"{module.key}_cf")
    c = SH.card(P, f"{module.key}_dim", "Dimensionality",
                "PCA before fitting reduces noise and speeds up distance methods with many features.")
    n_num = len(ud.numeric) + sum(ud.X[x].nunique() for x in ud.categorical)
    pca_on = c.checkbox("Reduce with PCA", value=bool(S.u.get("setup", {}).get("pca")) or n_num > 100,
                        key=f"{module.key}_pcaon")
    pca_n = int(c.number_input("Components", 2, max(2, min(100, n_num, len(ud.X) - 1)),
                               min(10, max(2, n_num - 1)), key=f"{module.key}_pcan", disabled=not pca_on))
    extra = {}
    if module.key == "anomaly":
        c = SH.card(P, f"{module.key}_cont", "Contamination", "Expected share of anomalies; sets the flag threshold.")
        guess = float(np.clip(np.mean(ud.reference), 0.01, 0.3)) if ud.reference is not None else 0.05
        extra["contamination"] = float(c.slider("Fraction", 0.01, 0.3, round(guess, 2), 0.01, key=f"{module.key}_contam"))
    cfg = PreprocessConfig(numeric_imputation=imp, categorical_imputation=cimp, transform=tr, log_offset=offset,
                           scaling=sc, max_missing_fraction=None if mm >= 1 else mm,
                           variance_threshold=0.0 if vt else None, correlation_threshold=cf if cf_on else None)
    if problem:
        P.error(problem)
    if SH.run_button(P, module, "Confirm setup", disabled=bool(problem), key_suffix="_setup"):
        clear(S, "setup")
        pre = build_unsupervised_preprocessor(ud.numeric, ud.categorical, cfg, pca_n if pca_on else None)
        try:
            Z = pre.fit_transform(ud.X)
        except Exception as e:  # noqa: BLE001
            main.error(f"Preprocessing failed: {type(e).__name__}: {e}"); return
        S.u.update(Z=pd.DataFrame(np.asarray(Z, float), index=ud.X.index), pre=pre, cfg=cfg,
                   setup={"pca": pca_n if pca_on else None, **extra})
        S.log("Setup", f"{Z.shape[1]} model dimensions" + (f", PCA {pca_n}" if pca_on else ""))
        SH.advance(module, "compare")
        st.rerun()

    SH.section(main, "Preprocessing pipeline", "No outcome, so no train/test split: the pipeline is fitted on all "
                                               "samples. Results are judged by internal criteria, resampling stability "
                                               "and, when you have one, the reference label.")
    steps = [ln.split(":")[0] if len(ln) > 46 else ln for ln in describe_preprocessing(cfg)[:-1]]
    SH.pipeline_strip(main, steps + ([f"PCA ({pca_n})"] if pca_on else []) + ["algorithm"])
    Z = S.u.get("Z")
    if Z is None:
        main.info("Review the settings in the panel and confirm them.")
        return
    main.success(f"Setup confirmed: {Z.shape[0]:,} samples × {Z.shape[1]} model dimensions.")
    pca = S.u["pre"].named_steps.get("pca") if hasattr(S.u["pre"], "named_steps") else None
    if pca is not None:
        import plotly.graph_objects as go
        ev = pca.explained_variance_ratio_
        fig = go.Figure(go.Bar(x=[f"PC{i + 1}" for i in range(len(ev))], y=ev, marker_color="#7A4FB5" if module.key ==
                               "clustering" else "#B23A48"))
        fig.add_trace(go.Scatter(x=[f"PC{i + 1}" for i in range(len(ev))], y=np.cumsum(ev), mode="lines+markers",
                                 name="cumulative", line=dict(color="#16324F")))
        fig.update_layout(title=f"Variance explained by {len(ev)} components: {ev.sum():.0%}", height=320,
                          showlegend=False, yaxis_tickformat=".0%")
        fig_out(main, fig)
