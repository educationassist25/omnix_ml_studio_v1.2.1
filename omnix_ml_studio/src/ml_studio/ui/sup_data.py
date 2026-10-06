"""Supervised workflow — step 1 (Data) and step 2 (Setup)."""
from __future__ import annotations

import os
from dataclasses import replace

import numpy as np
import pandas as pd
import streamlit as st

from ..core.config import PreprocessConfig, StudyDesign, ValidationConfig
from ..core.data_validation import (MAX_UPLOAD_MB, DataLoadError, apply_exclusions, assess_quality, infer_task,
                                    infer_types, load_table, suggest_roles, tcga_info)
from ..core.feature_selection import METHODS, PERMUTATION_MAX_FEATURES, selection_stability, univariate_statistics
from ..core.preprocessing import check_log_validity, describe_preprocessing, prepare_xy
from ..core.splitting import holdout_split, recommend_strategy
from ..core.validation import HoldoutLedger
from ..visualization import feature_plots as FP
from ..visualization import xai_plots as XP
from . import shell as SH
from .components import downloads, safe_slider, fig_out

EXAMPLES = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "examples"))
NONE = "(none)"
POSITIVE_WORDS = {"1", "yes", "y", "true", "case", "positive", "pos", "disease", "event", "responder", "malignant",
                  "relapse", "death", "dead", "died", "tumor", "tumour", "cancer", "abnormal", "high"}
SEV = {"critical": "🔴", "warning": "🟠", "info": "🔵"}
EXAMPLE_FOR = {"classification": ["binary_classification.csv", "clinical_plus_800_genes.csv", "multiclass_classification.csv",
                                  "grouped_omics.csv", "omics_hdlss.csv", "batch_confounded.csv",
                                  "external_development.csv"],
               "regression": ["regression.csv", "clinical_plus_800_genes.csv"]}
STRATS = {"stratified_kfold": "Stratified K-fold", "kfold": "K-fold",
          "repeated_stratified_kfold": "Repeated stratified K-fold", "repeated_kfold": "Repeated K-fold",
          "group_kfold": "Group K-fold", "stratified_group_kfold": "Stratified group K-fold"}


# =========================================================================== DATA
def data_step(S, module, main, side):
    P = SH.panel(side, "data")
    c = SH.card(P, f"{module.key}_src", "Dataset", f"CSV, TSV or Excel up to {MAX_UPLOAD_MB} MB. One row per sample.")
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
    if S.df is None:
        SH.empty(main, "No dataset yet", "Upload a file or load an example in the panel on the right. "
                 "Each row should be one sample; columns are features, the outcome, and optional ID, "
                 "patient, batch or time columns.")
        return
    df = S.df
    roles_card(S, module, P, df)

    # ---- main: preview
    SH.kpis(main, [("rows", f"{df.shape[0]:,}"), ("columns", f"{df.shape[1]:,}"),
                   ("missing cells", f"{df.isna().values.mean():.1%}"), ("file", S.filename)])
    with main.expander("Preview", expanded=S.design is None):
        ncol = min(df.shape[1], 40)
        st.dataframe(df.iloc[:30, :ncol], width="stretch", height=260)
        if df.shape[1] > ncol:
            st.caption(f"Showing 30 rows and the first {ncol} of {df.shape[1]} columns.")
    if S.design is None:
        main.info("Assign the column roles in the panel, then confirm them to run the quality checks.")
        return
    quality_section(S, module, main)
    if S.data is not None:
        explore_section(S, module, main)


def roles_card(S, module, P, df):
    c = SH.card(P, f"{module.key}_roles", "Column roles",
                "Only the outcome and features are used for modelling.")
    cols = list(df.columns)
    d0 = S.design
    opt = [NONE] + cols
    ix = lambda v: opt.index(v) if v in opt else 0  # noqa: E731
    is_reg = module.key == "regression"
    # column analysis once per table (10,000+ columns made every click on this step slow)
    ck = (id(df), df.shape, is_reg)
    cache = st.session_state.get(f"{module.key}_rolecache")
    if not cache or cache[0] != ck:
        g = suggest_roles(df)
        nun = df.nunique(dropna=True)
        numeric = {x for x in cols if pd.api.types.is_numeric_dtype(df[x])}
        tgt = [x for x in cols if ((x in numeric and nun[x] > 10) == is_reg) or (not is_reg and nun[x] <= 20)]
        id_like = {x for x in cols if nun[x] >= 0.95 * len(df) and not
                   (is_reg and pd.api.types.is_float_dtype(df[x]))} | ({g["sample_id"]} - {None})
        tgt = [x for x in tgt if x not in id_like] or tgt or cols
        st.session_state[f"{module.key}_rolecache"] = cache = (ck, g, tgt)
    g, tgt_opts = cache[1], cache[2]
    guess = d0.target if d0 else (g["target"] if g["target"] in tgt_opts else tgt_opts[-1])
    target = c.selectbox("Outcome", tgt_opts, index=tgt_opts.index(guess) if guess in tgt_opts else 0,
                         key=f"{module.key}_target",
                         help="Categorical outcome (≤ 20 levels)" if not is_reg else "Numeric outcome")
    task = "regression" if is_reg else infer_task(df[target].astype(str))
    if not is_reg:
        c.caption(f"{df[target].nunique()} classes: {'binary' if task == 'binary' else 'multiclass'} classification")
    pos = None
    if task == "binary":
        lv = sorted(df[target].dropna().astype(str).unique())
        likely = [i for i, v in enumerate(lv) if v.strip().lower() in POSITIVE_WORDS]
        pos = c.selectbox("Positive class", lv, index=likely[0] if likely else len(lv) - 1, key=f"{module.key}_pos",
                          help="Sensitivity, PPV and probabilities refer to this class.")
    pend = st.session_state.pop(f"{module.key}_grp_pending", None)
    gver = st.session_state.get(f"{module.key}_grp_ver", 0)
    if pend and pend in cols:                                 # a new key makes the browser build a fresh selector
        gver += 1
        st.session_state[f"{module.key}_grp_ver"] = gver
    a, b = c.columns(2)
    sid = a.selectbox("Sample ID", opt, index=ix(d0.sample_id if d0 else g["sample_id"]), key=f"{module.key}_sid")
    group = b.selectbox("Patient / group", opt, index=ix(pend if pend in cols else (d0.group if d0 else g["group"])),
                        key=f"{module.key}_grp" + (f"_{gver}" if gver else ""),
                        help="Repeated measures from one subject stay together in every split.")
    if sid != NONE and group == NONE:
        info = tcga_info(df[sid])
        if info and info["n_multi"]:
            c.warning(f"'{sid}' holds TCGA barcodes: {info['n_samples']} samples from {info['n_patients']} patients, "
                      f"and {info['n_multi']} patients contribute more than one sample (e.g. tumour and matched "
                      "normal). Group by patient so one patient's samples never sit on both sides of a split.")
            if c.button("Add patient column and group by it", key=f"{module.key}_tcga", use_container_width=True):
                S.df = S.df.assign(tcga_patient=info["patient"].values)
                st.session_state[f"{module.key}_grp_pending"] = "tcga_patient"
                S.log("Roles", f"patient derived from TCGA barcode ({info['n_patients']} patients)")
                st.rerun()
    a, b = c.columns(2)
    batch = a.selectbox("Batch", opt, index=ix(d0.batch if d0 else g["batch"]), key=f"{module.key}_batch",
                        help="Used for confounding checks only — never as a predictor.")
    time_c = b.selectbox("Time point", opt, index=ix(d0.time if d0 else g["time"]), key=f"{module.key}_time")
    roles = {target, sid, group, batch, time_c} - {NONE}
    cand = [x for x in cols if x not in roles]
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
    c.caption(f"{len(feats)} features: {len(num)} numeric, {len(cat)} categorical")
    with c.expander("Types and covariates"):
        cat_over = st.multiselect("Treat as categorical", num, key=f"{module.key}_catover",
                                  help="e.g. integer-coded stage")
        covs = st.multiselect("Clinical covariates (for transforms and reporting)", feats, key=f"{module.key}_covs")
    if SH.run_button(P, module, "Confirm roles and check quality", disabled=not feats, key_suffix="_roles"):
        S.invalidate("design")
        S.design = StudyDesign(target=target, task=task, features=feats, sample_id=None if sid == NONE else sid,
                               group=None if group == NONE else group, batch=None if batch == NONE else batch,
                               time=None if time_c == NONE else time_c, covariates=covs, categorical=cat_over,
                               positive_label=pos)
        st.session_state.pop(f"{module.key}_qcache", None)
        S.log("Roles", f"{target} ({task}), {len(feats)} features")
        st.rerun()


def quality_section(S, module, main):
    d = S.design
    key = (id(S.df), repr(d.to_dict()))
    cache = st.session_state.get(f"{module.key}_qcache")
    if cache and cache[0] == key:
        rep = cache[1]
    else:
        with main.spinner("Running data-quality checks…"):
            rep = assess_quality(S.df, d)
        st.session_state[f"{module.key}_qcache"] = (key, rep)
    SH.section(main, "Data quality", "Nothing is removed until you confirm. Critical integrity problems must be "
                                     "resolved; scientific warnings need an explicit acknowledgement.")
    SH.kpis(main, [("samples", rep.n_samples), ("features", rep.n_features),
                   ("numeric / categorical", f"{rep.n_numeric} / {rep.n_categorical}"),
                   ("critical issues", len(rep.critical))])
    for sev in ("critical", "warning", "info"):
        for i in [x for x in rep.issues if x.severity == sev]:
            with main.expander(f"{SEV[sev]}  {i.message}", expanded=False):
                if i.columns:
                    st.write("Columns:", ", ".join(map(str, i.columns[:60])) + (" …" if len(i.columns) > 60 else ""))
                if i.rows:
                    ids = S.df.loc[i.rows[:60], d.sample_id].tolist() if d.sample_id else i.rows[:60]
                    st.write("Rows:", ", ".join(map(str, ids)) + (" …" if len(i.rows) > 60 else ""))
    sug = rep.suggested_feature_exclusions
    # integrity problems are pre-selected; strong single-feature separators are NOT (often real biology)
    pre = [x for x, why in sug.items() if x in d.features and why != "possible target leakage"]
    leak = [x for x, why in sug.items() if x in d.features and why == "possible target leakage"]
    if leak:
        main.info(f"{len(leak)} feature(s) alone separate the outcome almost perfectly (e.g. {', '.join(leak[:5])}). "
                  "This can be leakage (measured after, or derived from, the outcome) or a genuinely strong "
                  "marker, as is common for tumour versus normal tissue. They are kept unless you exclude them.")
        if main.checkbox(f"Exclude all {len(leak)} of them", key=f"{module.key}_dropleak"):
            pre = pre + leak
    drop_f = main.multiselect("Features to exclude", d.features, default=pre,
                              format_func=lambda x: f"{x} — {sug[x]}" if x in sug else x,
                              key=f"{module.key}_dropf_{len(pre)}")
    miss_y = [r for r, why in rep.suggested_sample_exclusions.items() if why == "missing target"]
    flagged = sorted({r for i in rep.issues if i.code in ("sample_missing", "duplicate_ids") for r in i.rows})
    lab = (lambda r: f"{S.df.loc[r, d.sample_id]} (row {r})") if d.sample_id else (lambda r: f"row {r}")
    drop_r = main.multiselect("Samples to exclude", flagged, format_func=lab, key=f"{module.key}_dropr") if flagged else []
    if miss_y:
        main.caption(f"{len(miss_y)} sample(s) without an outcome will be excluded from modelling.")

    def resolved(i):
        if i.columns and set(i.columns) <= set(drop_f):
            return True
        if i.code == "duplicate_ids" and d.sample_id:
            return not S.df.drop(index=[r for r in drop_r if r in S.df.index])[d.sample_id].duplicated().any()
        return bool(i.rows) and set(i.rows) <= set(drop_r)
    ACK = {"target_leakage": "I have reviewed the strongly separating features: they are measured before, and not derived "
                             "from, the outcome (or I excluded them).",
           "batch_confounding": "Batch is confounded with the outcome; I will interpret results accordingly."}
    open_crit = [i for i in rep.critical if not resolved(i)]
    blocking = [i for i in open_crit if i.code not in ACK]
    ok = True
    for i in open_crit:
        if i.code in ACK:
            ok &= main.checkbox(ACK[i.code], key=f"{module.key}_ack_{i.code}")
    if blocking:
        main.error("Resolve before continuing: " + "; ".join(i.message for i in blocking))
    if main.button("Confirm exclusions and prepare data", key=f"run_{module.key}_prep", type="primary",
                   disabled=bool(blocking) or not ok):
        df2, d2, audit = apply_exclusions(S.df, d, drop_f, drop_r + miss_y)
        try:
            data = prepare_xy(df2, d2)
        except ValueError as e:
            main.error(str(e)); return
        S.invalidate("split")
        S.design, S.audit, S.data, S.quality = d2, audit, data, rep
        S.log("Quality", f"excluded {len(drop_f)} features, {len(drop_r) + len(miss_y)} samples")
        SH.advance(module, "setup")
        st.rerun()
    if S.data is not None:
        main.success(f"Data prepared: {len(S.data.y)} samples, {len(S.design.features)} features. "
                     "Continue to Setup.")


def explore_section(S, module, main):
    d, D = S.design, S.data
    df = S.df.loc[D.index]
    num = D.numeric
    SH.section(main, "Explore", "For orientation only. These views use all prepared samples and never feed into "
                                "modelling choices.")
    views = ["Outcome", "Missing values", "Feature distribution", "PCA and outliers", "Correlation", "Batch and groups"]
    v = SH.picker(main, f"{module.key}_explore_view", "View", views)
    if v == "Outcome":
        fig_out(main, FP.target_distribution(df[d.target], d.task))
    elif v == "Missing values":
        if D.X.isna().values.any():
            fig_out(main, FP.missingness_heatmap(D.X))
        else:
            main.success("No missing values in the prepared features.")
    elif v == "Feature distribution" and num:
        f = main.selectbox("Feature", num, key=f"{module.key}_eda_feat")
        by = d.target if d.task != "regression" else d.batch
        if by:
            fig_out(main, FP.feature_box(df, f, by))
        fig_out(main, FP.variance_summary(D.X[num]))
    elif v == "PCA and outliers" and len(num) >= 2:
        from sklearn.decomposition import PCA
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        nonneg = (np.nan_to_num(D.X[num].to_numpy(float)) >= 0).all()
        log = main.checkbox("log2(x + 1) first", value=bool(nonneg and len(num) > 50), disabled=not nonneg,
                            key=f"{module.key}_pca_log")
        V = D.X[num].to_numpy(float)
        V = np.log2(np.clip(V, 0, None) + 1) if log else V
        V = SimpleImputer(strategy="median").fit_transform(V)
        V = StandardScaler().fit_transform(V[:, V.std(0) > 0])
        p = PCA(min(10, V.shape[1], V.shape[0] - 1), random_state=0)
        sc = p.fit_transform(V)
        cby = main.selectbox("Colour by", [c for c in (d.target, d.batch, d.group, d.time) if c], key=f"{module.key}_pca_col")
        hov = df[d.sample_id].astype(str).values if d.sample_id else None
        fig_out(main, FP.pca_scatter(sc, p.explained_variance_ratio_, df[cby],
                                         "median-imputed, " + ("log2, " if log else "") + "standardised", hov))
        fig, flag = FP.outlier_scores(sc, hov if hov is not None else df.index.astype(str))
        fig_out(main, fig)
    elif v == "Correlation" and len(num) >= 2:
        k = safe_slider(main, "Features (highest variance)", min(10, len(num)), min(150, len(num)), 40, key=f"{module.key}_corr_k")
        fig_out(main, FP.correlation_heatmap(D.X[num], k))
    elif v == "Batch and groups":
        if d.batch:
            main.dataframe(pd.crosstab(df[d.batch], df[d.target]) if d.task != "regression"
                           else df.groupby(d.batch)[d.target].describe(), width="stretch")
        if d.group:
            sz = df[d.group].value_counts()
            main.markdown(f"{len(sz)} groups; rows per group: median {sz.median():.0f}, maximum {sz.max()}.")
        if not d.batch and not d.group:
            main.caption("No batch or group column was assigned.")


# =========================================================================== SETUP
def setup_step(S, module, main, side):
    d, D = S.design, S.data
    P = SH.panel(side, "setup")
    p0, v0 = S.prep_cfg, S.val_cfg

    c = SH.card(P, f"{module.key}_imp", "Missing values")
    imp = c.selectbox("Numeric", ["median", "mean", "knn", "constant", "none"],
                      index=["median", "mean", "knn", "constant", "none"].index(p0.numeric_imputation), key=f"{module.key}_imp")
    fill = c.number_input("Fill value", value=float(p0.numeric_fill_value), key=f"{module.key}_fill") if imp == "constant" else 0.0
    cimp = c.selectbox("Categorical", ["most_frequent", "constant"],
                       index=["most_frequent", "constant"].index(p0.categorical_imputation), key=f"{module.key}_cimp",
                       format_func=lambda x: "most frequent" if x == "most_frequent" else "'missing' category")

    c = SH.card(P, f"{module.key}_tr", "Transformation and scaling")
    TR = ["none", "log2", "log10", "yeo-johnson", "quantile"]
    tr = c.selectbox("Transformation", TR, index=TR.index(p0.transform), key=f"{module.key}_tr")
    offset, t_cols, problem = None, None, None
    if tr != "none" and D.numeric:
        covs = [x for x in D.numeric if x in set(d.covariates)]
        scope = c.radio("Apply to", ["All numeric", "All except clinical covariates", "Chosen columns"],
                        key=f"{module.key}_tscope", index=0 if p0.transform_columns is None else 2)
        if scope == "All except clinical covariates":
            t_cols = [x for x in D.numeric if x not in set(covs)]
            if not covs:
                c.caption("No clinical covariates were marked in the Data step.")
        elif scope == "Chosen columns":
            t_cols = c.multiselect("Columns", D.numeric, default=[x for x in (p0.transform_columns or D.numeric)
                                                                   if x in D.numeric], key=f"{module.key}_tcols")
            if not t_cols:
                problem = "choose at least one column to transform."
    if tr in ("log2", "log10") and D.numeric:
        cols_chk = t_cols if t_cols is not None else D.numeric
        chk = check_log_validity(D.X, cols_chk) if cols_chk else {"valid_without_offset": True}
        if not chk["valid_without_offset"]:
            c.warning(f"{chk['n_zero']} zero and {chk['n_negative']} negative values, e.g. in "
                      f"{', '.join(chk['features_with_nonpositive'][:3])}. log(0) is undefined.")
            if c.checkbox("Add an explicit offset (x + c)", value=p0.log_offset is not None, key=f"{module.key}_offon"):
                offset = c.number_input("c", value=float(p0.log_offset or 1.0), min_value=0.0, format="%.4g",
                                        key=f"{module.key}_off")
                vmin = float(np.nanmin(D.X[cols_chk].to_numpy(float)))
                if offset + vmin <= 0:
                    problem = f"offset {offset:g} leaves non-positive values (minimum {vmin:g})."
            else:
                problem = "the log transform needs an explicit offset, fewer columns, or Yeo-Johnson."
    SC = ["standard", "robust", "minmax", "none"]
    sc = c.selectbox("Scaling", SC, index=SC.index(p0.scaling), key=f"{module.key}_sc")

    c = SH.card(P, f"{module.key}_flt", "Feature filtering", "Every filter is refit inside each training fold.")
    mm = c.slider("Drop features missing in more than", 0.05, 1.0, p0.max_missing_fraction or 1.0, 0.05,
                  key=f"{module.key}_mm")
    vt = c.checkbox("Drop constant features", value=p0.variance_threshold is not None, key=f"{module.key}_vt")
    cf_on = c.checkbox("Correlation filter", value=p0.correlation_threshold is not None, key=f"{module.key}_cfon")
    cf = c.slider("|r| above", 0.7, 0.99, p0.correlation_threshold or 0.95, 0.01, disabled=not cf_on, key=f"{module.key}_cf")
    wide = len(d.features) > 2000
    uk_on = c.checkbox("Univariate pre-filter (top k)", value=bool(p0.univariate_k) or (wide and not S.design_confirmed),
                       key=f"{module.key}_ukon",
                       help="Keeps the k features most associated with the outcome, chosen again inside every "
                            "training fold (no leakage). Strongly recommended above ~2,000 features: every model "
                            "becomes many times faster and lighter.")
    uk = c.number_input("k", 2, max(2, len(d.features)),
                        int(p0.univariate_k or (min(1000, len(d.features)) if wide else min(100, len(d.features)))),
                        disabled=not uk_on, key=f"{module.key}_uk")
    if wide and not uk_on:
        c.warning(f"All {len(d.features):,} features will reach every model: expect long run times.")
    fs_opts = [None] + list(METHODS)
    fs_m = c.selectbox("Feature selection", fs_opts, index=fs_opts.index(p0.fs_method) if p0.fs_method in fs_opts else 0,
                       format_func=lambda m: "none" if m is None else METHODS[m], key=f"{module.key}_fsm")
    fs_k = c.number_input("Features to keep", 1, max(1, len(d.features)), int(min(p0.fs_k, len(d.features))),
                          disabled=fs_m in (None, "rfecv"), key=f"{module.key}_fsk")
    imb = "none"
    if d.task != "regression":
        IMB = ["none", "class_weight", "oversample", "undersample", "smote"]
        imb = c.selectbox("Class imbalance", IMB, index=IMB.index(p0.imbalance), key=f"{module.key}_imb",
                          help="Resampling runs on training folds only.")
    new_p = PreprocessConfig(numeric_imputation=imp, numeric_fill_value=fill, categorical_imputation=cimp, transform=tr,
                             log_offset=offset, transform_columns=t_cols, scaling=sc,
                             max_missing_fraction=None if mm >= 1.0 else mm, variance_threshold=0.0 if vt else None,
                             correlation_threshold=cf if cf_on else None, univariate_k=int(uk) if uk_on else None,
                             fs_method=fs_m, fs_k=int(fs_k), imbalance=imb)

    n_units = len(np.unique(D.groups)) if D.groups is not None else len(D.y)
    min_class = int(np.bincount(D.y).min()) if d.task != "regression" else None
    rs_, rk, rr, why = recommend_strategy(d.task, n_units, D.groups is not None, min_class, bool(d.time))
    c = SH.card(P, f"{module.key}_val", "Validation design")
    allowed = [k for k in STRATS if (k in ("group_kfold", "stratified_group_kfold")) == (D.groups is not None)
               and not (d.task == "regression" and "stratified" in k)]
    default = v0.cv_strategy if (S.design_confirmed and v0.cv_strategy in allowed) else (rs_ if rs_ in allowed else allowed[0])
    strat = c.selectbox("Cross-validation", allowed, index=allowed.index(default), format_func=STRATS.get,
                        key=f"{module.key}_strat")
    a, b = c.columns(2)
    k = a.number_input("Folds", 2, 20, int(v0.n_folds if S.design_confirmed else rk), key=f"{module.key}_k")
    rep = b.number_input("Repeats", 1, 20, int(v0.n_repeats if S.design_confirmed else rr), key=f"{module.key}_rep",
                         disabled="repeated" not in strat and "group" not in strat)
    train_pct = c.slider("Training set (development)", 60, 100, int(round((1 - float(v0.holdout_fraction)) * 100)), 5,
                         format="%d%%", key=f"{module.key}_trainpct",
                         help="Share of samples used to compare, tune and fit models. The rest is the independent "
                              "holdout (test set): split once, sealed now, and used only in the Validate step. "
                              "100% = no holdout (use nested CV).")
    hf = round(1 - train_pct / 100, 2)
    unit = "patients" if D.groups is not None else "samples"
    try:   # exact counts of the split that confirming would make
        _pv = holdout_split(D.y, d.task, hf, D.groups, int(st.session_state.get(f"{module.key}_seed", v0.random_state)))
        te_idx = np.asarray(_pv.test_idx)
        n_test = len(np.unique(D.groups[te_idx])) if D.groups is not None else len(te_idx)
    except Exception:  # noqa: BLE001
        n_test = int(round(n_units * hf))
    c.markdown(f'<div class="omx-split"><span class="tr">Training set<b>{n_units - n_test:,}</b>{unit} · {train_pct}%</span>'
               f'<span class="te">Independent holdout (test set)<b>{n_test:,}</b>{unit} · {100 - train_pct}%</span></div>',
               unsafe_allow_html=True)
    with c.expander("Advanced"):
        inner = st.number_input("Nested CV inner folds", 2, 10, int(v0.inner_folds), key=f"{module.key}_inner")
        seed = st.number_input("Random seed", 0, 2 ** 31 - 1, int(v0.random_state), key=f"{module.key}_seed")
        nboot = st.number_input("Bootstrap resamples", 200, 5000, int(v0.n_bootstrap), 100, key=f"{module.key}_nboot")
        from ..core.runner import available_cpus
        jopts = [0, 1, 2, 4, 8, 16]
        jobs = st.selectbox("CPU cores to use", jopts, index=jopts.index(v0.n_jobs) if S.design_confirmed and
                            v0.n_jobs in jopts else 0, key=f"{module.key}_jobs",
                            format_func=lambda j: f"Auto (all {available_cpus()} available)" if j == 0 else str(j),
                            help="Models are compared in parallel, one per core, each in a protected process with "
                                 "its own memory limit.")
    new_v = ValidationConfig(hf, strat, int(k), int(rep) if ("repeated" in strat or "group" in strat) else 1,
                             int(inner), int(seed), int(nboot), int(jobs))
    ok = P.checkbox("This design matches the study (independent units, repeated measures, intended use).",
                    key=f"{module.key}_confirm")
    if problem:
        P.error(f"Cannot confirm yet: {problem}")
    if SH.run_button(P, module, "Confirm setup and seal holdout", disabled=not ok or bool(problem), key_suffix="_setup"):
        S.invalidate("split")
        S.prep_cfg, S.val_cfg = new_p, new_v
        S.split = holdout_split(D.y, d.task, hf, D.groups, int(seed))
        S.ledger = HoldoutLedger()
        S.design_confirmed = True
        S.log("Setup", f"{STRATS[strat]} k={k}×{new_v.n_repeats}; {S.split.description}")
        SH.advance(module, "compare")
        st.rerun()

    # ---- main
    SH.section(main, "Preprocessing pipeline", "Fitted inside every training fold, in this order.")
    SH.pipeline_strip(main, [ln.split(":")[0] if len(ln) > 46 else ln for ln in describe_preprocessing(new_p)[:-1]]
                      + ["model"])
    SH.section(main, "Validation design")
    main.info(f"Recommended for these data: {STRATS[rs_]}, {rk} folds" + (f" × {rr} repeats" if rr > 1 else "")
              + f". {why}")
    validation_diagram(main, n_units, hf, int(k), new_v.n_repeats, D.groups is not None)
    if hf > 0 and n_units * hf < 20:
        main.warning(f"The holdout would hold only ~{int(n_units * hf)} independent units; its estimate will be "
                     "imprecise. With small data, consider no holdout and rely on nested CV.")
    if S.design_confirmed:
        main.success(f"Setup confirmed. Holdout sealed: {S.split.description}.")
        if new_p != S.prep_cfg or new_v != S.val_cfg:
            main.warning("The panel differs from the confirmed setup. Confirming again resets model results.")
        stability_section(S, module, main)


def validation_diagram(main, n, hf, k, rep, grouped):
    import plotly.graph_objects as go
    dev = 1 - hf
    fig = go.Figure()
    for i in range(k):
        x0 = dev * i / k
        fig.add_shape(type="rect", x0=0, x1=dev, y0=k - i - .38, y1=k - i + .38, fillcolor="#E6ECF2", line_width=0)
        fig.add_shape(type="rect", x0=x0, x1=x0 + dev / k, y0=k - i - .38, y1=k - i + .38, fillcolor="#3B4CCA",
                      opacity=.75, line_width=0)
    if hf > 0:
        fig.add_shape(type="rect", x0=dev, x1=1, y0=.5, y1=k + .5, fillcolor="#B23A48", opacity=.18, line_width=0)
        fig.add_annotation(x=dev + hf / 2, y=(k + 1) / 2, text=f"test set<br>(sealed holdout)<br>{hf:.0%}", showarrow=False,
                           font=dict(color="#B23A48"))
    fig.update_xaxes(range=[0, 1], tickformat=".0%", title="training set (left) and test set (right), share of samples" + (" (grouped by patient)" if grouped else ""))
    fig.update_yaxes(tickvals=list(range(1, k + 1)), ticktext=[f"fold {k - i}" for i in range(k)], range=[.4, k + .6])
    fig.update_layout(height=70 + 34 * k, margin=dict(l=60, r=20, t=30, b=40), showlegend=False,
                      title=f"{k}-fold CV on the development set" + (f", repeated {rep}×" if rep > 1 else "")
                            + "; dark = validation fold")
    fig_out(main, fig)


def stability_section(S, module, main):
    d, D, tr = S.design, S.data, S.split.train_idx
    p = len(d.features)
    SH.section(main, "Feature selection stability (optional)",
               "Compare selection methods across resamples of the development set. Refits preprocessing and "
               "selection on each training set only.")
    with main.container(border=True):
        big = p > PERMUTATION_MAX_FEATURES and not S.prep_cfg.univariate_k
        opts = [m for m in METHODS if not (m == "permutation" and big)]
        a, b, c = st.columns([3, 1, 1])
        methods = a.multiselect("Methods", opts, default=["univariate", "l1", "tree"], format_func=METHODS.get,
                                key=f"{module.key}_fs_methods")
        k = int(b.number_input("k", 1, max(1, p), max(1, min(20, p // 4)), key=f"{module.key}_fs_kk"))
        reps = int(c.number_input("Repeats", 1, 10, 3, key=f"{module.key}_fs_reps"))
        if k >= p:
            st.warning("k keeps every feature, so stability would be trivially perfect.")
        if st.button("Run stability analysis", key=f"{module.key}_fs_run", disabled=not methods):
            bar = st.progress(0.0)
            try:
                S.stability = selection_stability(D, tr, d.task, methods, k, S.prep_cfg, S.val_cfg, reps,
                                                  progress=lambda i, n, m: bar.progress(i / n, text=m))
                S.fstats = univariate_statistics(D, tr, d.task)
                S.log("Feature stability", f"{len(methods)} methods, k={k}")
            except Exception as e:  # noqa: BLE001
                st.error(f"Stability analysis failed: {e}")
            bar.empty()
    r = S.stability
    if r is None:
        return
    main.dataframe(r.indices.drop(columns=["method"]).round(3), hide_index=True, width="stretch")
    main.caption("Kuncheva 1 = identical selections, ≈ 0 = random. Stability measures consistency, not validity: "
                 "overlapping training sets can agree without real signal. Judge predictive value with nested CV.")
    view = SH.picker(main, f"{module.key}_fs_view", "View",
                     ["Selection frequency", "Predictive vs significant", "Method overlap", "Tables"])
    if view == "Selection frequency":
        fig_out(main, XP.stability_bars(r.frequency, 30))
    elif view == "Predictive vs significant":
        from ..core.feature_selection import to_original
        m = main.selectbox("Method", list(r.frequency.columns), format_func=METHODS.get, key=f"{module.key}_fs_sig")
        fo = r.frequency[m].groupby(lambda f: to_original(f, D.categorical)).max()
        fig_out(main, XP.predictive_vs_significant(fo, S.fstats, METHODS[m]))
        main.info("Selection frequency shows what a multivariable model relies on; univariate q-values test each "
                  "feature alone. Neither establishes a biomarker without independent validation.")
    elif view == "Method overlap" and len(r.overlap) > 1:
        fig_out(main, XP.overlap_heatmap(r.overlap, METHODS))
    elif view == "Tables":
        tab = r.frequency.add_prefix("freq_").join(r.mean_rank.add_prefix("rank_")).reset_index(names="feature")
        main.dataframe(tab.round(3), hide_index=True, height=300, width="stretch")
        downloads(main, tab, "feature_stability", f"{module.key}_fsdl", {"univariate_stats": S.fstats})
    u1, u2 = main.columns([3, 1], vertical_alignment="bottom")
    use = u1.selectbox("Use in the modelling pipeline", [None] + list(r.frequency.columns),
                       format_func=lambda x: "no selection" if x is None else METHODS[x], key=f"{module.key}_fs_use")
    if u2.button("Apply", key=f"{module.key}_fs_apply", use_container_width=True):
        keep = (S.stability, S.fstats)
        S.prep_cfg = replace(S.prep_cfg, fs_method=use, fs_k=k)
        S.invalidate("models")
        S.stability, S.fstats = keep
        S.log("Pipeline", f"feature selection = {METHODS.get(use, 'none')}")
        st.rerun()
