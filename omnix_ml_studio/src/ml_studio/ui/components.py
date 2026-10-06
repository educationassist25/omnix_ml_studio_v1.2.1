"""Reusable widgets: model picker grid, hyperparameter editor, downloads, metric tables."""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd
import streamlit as st

from ..reporting.export_utils import to_csv_bytes, to_xlsx_bytes

SPEED_NOTE = {"slow": "slow", "medium": ""}


def model_picker(parent, key: str, specs: dict, families: Iterable[str], default: list,
                 badge=None, columns: int = 2) -> list:
    """Checkbox grid grouped by family with per-family 'all' toggles and quick presets.

    ``specs``: key -> spec with ``.name``, ``.family``, ``.speed``; ``badge(spec) -> str``.
    """
    fams = [f for f in families if any(s.family == f for s in specs.values())]
    fams += sorted({s.family for s in specs.values()} - set(fams))
    state_key = f"{key}__sel"
    if state_key not in st.session_state:
        st.session_state[state_key] = set(k for k in default if k in specs)
    sel: set = st.session_state[state_key]
    b1, b2, b3, b4 = parent.columns(4)
    if b1.button("Default", key=f"{key}_rec", use_container_width=True, help="A balanced, quick set"):
        sel.clear(); sel.update(k for k in default if k in specs); _sync(key, specs, sel); st.rerun()
    if b2.button("All", key=f"{key}_all", use_container_width=True, help=f"All {len(specs)} algorithms"):
        sel.clear(); sel.update(specs); _sync(key, specs, sel); st.rerun()
    if b3.button("Fast", key=f"{key}_fast", use_container_width=True, help="Everything not marked slow"):
        sel.clear(); sel.update(k for k, s in specs.items() if s.speed != "slow"); _sync(key, specs, sel); st.rerun()
    if b4.button("Clear", key=f"{key}_none", use_container_width=True):
        sel.clear(); _sync(key, specs, sel); st.rerun()
    grid = parent.columns(columns)
    for i, fam in enumerate(fams):
        col = grid[i % columns]
        members = [k for k, s in specs.items() if s.family == fam]
        col.markdown(f'<div class="omx-card-h" style="margin-top:.4rem">{fam}</div>', unsafe_allow_html=True)
        for k in members:
            s = specs[k]
            extra = badge(s) if badge else ""
            note = ", ".join(x for x in (SPEED_NOTE.get(s.speed, ""), extra) if x)
            # The stored selection is the single source of truth. The widget key carries the current
            # state, so a checkbox whose widget state was dropped (navigation, background refresh)
            # is recreated from the selection instead of coming back unticked.
            on = k in sel
            v = col.checkbox(f"{s.name}" + (f"  ({note})" if note else ""), value=on, key=f"{key}_cb_{k}_{int(on)}",
                             help=getattr(s, "notes", "") or None)
            if v != on:
                (sel.add if v else sel.discard)(k)
    return [k for k in specs if k in sel]


def _sync(key, specs, sel):
    """Presets change the stored selection; the checkboxes follow it on the next run."""
    return None


def param_editor(parent, key: str, estimator, space: dict) -> dict:
    """Editable values for the tunable hyperparameters, defaults taken from the estimator."""
    params = estimator.get_params()
    out = {}
    for full in space:
        name = full.replace("model__", "")
        if name not in params:
            continue
        cur = params[name]
        dist = space[full]
        wk = f"{key}_{name}"
        if isinstance(dist, list):
            opts = list(dist)
            if cur not in opts:
                opts = [cur] + opts
            v = parent.selectbox(name, opts, index=opts.index(cur), key=wk, format_func=lambda o: str(o))
        elif isinstance(cur, bool):
            v = parent.checkbox(name, value=cur, key=wk)
        elif isinstance(cur, (int, np.integer)) and not isinstance(cur, bool):
            v = int(parent.number_input(name, value=int(cur), step=1, key=wk))
        elif isinstance(cur, (float, np.floating)):
            v = float(parent.number_input(name, value=float(cur), format="%.5g", key=wk))
        else:
            continue
        if v != cur:
            out[full] = v
    return out


def downloads(parent, df: pd.DataFrame, name: str, key: str, extra: Optional[dict] = None):
    c1, c2, _ = parent.columns([1, 1, 3])
    c1.download_button("Download CSV", to_csv_bytes(df), f"{name}.csv", "text/csv", key=f"{key}_csv",
                       use_container_width=True)
    c2.download_button("Download Excel", to_xlsx_bytes({name[:31]: df, **(extra or {})}), f"{name}.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key=f"{key}_xlsx",
                       use_container_width=True)


def ci_table(ci: pd.DataFrame) -> pd.DataFrame:
    d = ci.copy()
    d["95% CI"] = [f"{a:.3f} – {b:.3f}" if pd.notna(a) else "—" for a, b in zip(d["ci_low"], d["ci_high"])]
    return d[["label", "estimate", "95% CI"]].rename(columns={"label": "Metric", "estimate": "Estimate"}) \
        .round({"Estimate": 3})


def safe_slider(parent, label, lo, hi, value, step=1, key=None, **kw):
    lo, hi = int(lo), int(hi)
    if hi <= lo:
        parent.caption(f"{label}: {hi}")
        return hi
    return parent.slider(label, lo, hi, int(min(max(value, lo), hi)), step, key=key, **kw)


# --------------------------------------------------------------------------- figures
DPI_OPTIONS = [150, 300, 600, 1200]
WIDTH_OPTIONS = [3.5, 5.0, 7.0, 10.0, 14.0]


def page_figures() -> list:
    """Figures rendered so far in this script run (reset by ``reset_page_figures``)."""
    return st.session_state.setdefault("_page_figs", [])


def reset_page_figures():
    st.session_state["_page_figs"] = []


def fig_out(parent, fig, key: Optional[str] = None, name: Optional[str] = None, download: bool = True):
    """Render a Plotly figure with a download menu (PNG, JPEG, WebP, SVG, PDF, HTML at chosen DPI)."""
    from ..visualization.export import title_of
    from ..visualization.export import slug as _slug
    figs = page_figures()
    name = name or title_of(fig) or f"figure_{len(figs) + 1}"
    k = key or f"fig_{_slug(name)}_{len(figs)}"
    figs.append((name, fig))
    parent.plotly_chart(fig, width="stretch", key=k,
                        config={"displaylogo": False,
                                "toImageButtonOptions": {"format": "png", "scale": 4, "filename": _slug(name)}})
    if download:
        _download_menu(parent, fig, name, k)


def _download_menu(parent, fig, name, k):
    from ..visualization import export as EX
    with parent.popover("Download figure", icon=":material/download:"):
        fmt_label = st.selectbox("Format", list(EX.FORMATS), key=f"{k}_fmt")
        fmt = EX.FORMATS[fmt_label]
        dpi, width = 300, 7.0
        if fmt in EX.RASTER:
            dpi = st.select_slider("Resolution (DPI)", DPI_OPTIONS, value=300, key=f"{k}_dpi")
            width = st.select_slider("Print width (inches)", WIDTH_OPTIONS, value=7.0, key=f"{k}_w")
            px = int(dpi * width)
            st.caption(f"{px:,} px wide.")
        elif fmt in ("svg", "pdf"):
            st.caption("Vector format: sharp at any size; editable in Illustrator, Inkscape or PowerPoint.")
        else:
            st.caption("Interactive figure that opens in any browser (zoom, hover).")
        ok = fmt == "html" or EX.static_available()
        if not ok:
            st.warning("Image export needs Chrome/Chromium on the server. Use HTML, or the camera icon on the chart "
                       "(PNG at 4× screen resolution).")
        st.download_button(f"Download {fmt_label.split(' ')[0]}", data=lambda: EX.export(fig, fmt, dpi, width),
                           file_name=EX.filename(name, fmt), mime=EX.MIME[fmt], key=f"{k}_dl", disabled=not ok,
                           on_click="ignore", type="primary", use_container_width=True)


def download_all_figures(parent, key: str, figs: Optional[list] = None, label: str = "Download all figures"):
    """ZIP of every figure on the page in the chosen formats."""
    from ..visualization import export as EX
    figs = list(figs if figs is not None else page_figures())
    if not figs:
        return
    with parent.popover(f"{label} ({len(figs)})", icon=":material/folder_zip:"):
        fl = st.multiselect("Formats", list(EX.FORMATS), default=["PNG", "SVG", "PDF"], key=f"{key}_fmts")
        dpi = st.select_slider("Resolution for PNG / JPEG / WebP (DPI)", DPI_OPTIONS, value=300, key=f"{key}_dpi")
        width = st.select_slider("Print width (inches)", WIDTH_OPTIONS, value=7.0, key=f"{key}_w")
        ok = EX.static_available() or fl == ["HTML (interactive)"]
        st.caption(f"{len(figs)} figure(s) × {len(fl)} format(s). Building the archive takes about "
                   f"{max(1, round(0.6 * len(figs) * len(fl)))} s.")
        st.download_button("Download ZIP", data=lambda: EX.export_zip(figs, fl, dpi, width),
                           file_name=f"{key}_figures.zip", mime="application/zip", key=f"{key}_zip",
                           disabled=not fl or not ok, on_click="ignore", type="primary", use_container_width=True)


TOP_N = [10, 15, 20, 25, 30, 40, 50, 75, 100]


def feature_filter(parent, key: str, features: list, covariates: list | None = None, default_n: int = 20):
    """Feature group (all / omics / clinical) and Top N. Returns (allowed set or None, n, group label)."""
    from ..core.feature_groups import split_feature_groups
    g = split_feature_groups(features, covariates)
    opts = ["All features"]
    if g["omics"] and g["clinical"]:
        opts += [f"Genes / omics ({len(g['omics'])})", f"Clinical ({len(g['clinical'])})"]
    grp = parent.selectbox("Features to rank", opts, key=f"{key}_grp",
                           help="Rank only genes (omics) or only clinical variables. " + g["source"].capitalize() + ".")
    n = parent.select_slider("Show top", TOP_N, value=default_n, key=f"{key}_topn")
    if len(opts) > 1:
        with parent.expander(f"Clinical variables ({len(g['clinical'])})"):
            st.caption("Grouped by " + g["source"] + ".")
            st.write(", ".join(g["clinical"]))
    if grp.startswith("Genes"):
        return set(g["omics"]), n, "genes / omics"
    if grp.startswith("Clinical"):
        return set(g["clinical"]), n, "clinical variables"
    return None, n, "features"


# --------------------------------------------------------------------------- protected heavy work
HINT = {"memory": "Not enough memory for this model on these data. Use a univariate pre-filter in Setup (e.g. the "
                  "top 1,000 features), choose a lighter model, or run the app on a larger server.",
        "timeout": "It exceeded the time limit. Use a univariate pre-filter in Setup, fewer iterations, or a faster "
                   "model.",
        "crashed": "The worker process stopped (most often for lack of memory). The app itself is unaffected."}


def protected(S, parent, fn, label: str, timeout_s: Optional[float] = None, idx=None, **kw):
    """Run a heavy task in an isolated worker; show a clear message instead of crashing. Returns None on failure."""
    from ..core.runner import RunnerError, run_one
    payload = {"data": S.data, "idx": S.split.train_idx if idx is None else idx, "task": S.design.task,
               "prep": S.prep_cfg, "val": S.val_cfg}
    with parent.spinner(label):
        try:
            return run_one(payload, fn, timeout_s=timeout_s, **kw)
        except RunnerError as e:
            parent.error(f"{label.rstrip('…. ')} did not finish: {e}. {HINT.get(e.status, '')}")
            return None
