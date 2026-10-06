"""Feature-selection and explainability figures (Plotly, no Streamlit)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import MUTED, PALETTE


def stability_bars(freq: pd.DataFrame, n: int = 30) -> go.Figure:
    d = freq.head(n).iloc[::-1]
    fig = go.Figure()
    for i, m in enumerate(freq.columns):
        fig.add_trace(go.Bar(x=d[m], y=d.index, orientation="h", name=m, marker_color=PALETTE[i % len(PALETTE)]))
    fig.update_layout(barmode="group", title=f"Selection frequency across resamples — top {len(d)} features",
                      xaxis_title="fraction of resamples selected", xaxis_range=[0, 1.02],
                      height=160 + 18 * len(d) * max(1, len(freq.columns) // 2), margin=dict(l=160))
    return fig


def overlap_heatmap(ov: pd.DataFrame, labels: dict) -> go.Figure:
    lab = [labels.get(m, m) for m in ov.index]
    fig = go.Figure(go.Heatmap(z=ov.values, x=lab, y=lab, zmin=0, zmax=1, colorscale="Blues",
                               text=np.round(ov.values, 2), texttemplate="%{text}"))
    fig.update_layout(title="Overlap of consensus top-k sets between methods (Jaccard)",
                      height=200 + 45 * len(lab), yaxis_autorange="reversed")
    return fig


def predictive_vs_significant(freq: pd.Series, stats: pd.DataFrame, method_label: str) -> go.Figure:
    s = stats.set_index("feature")
    common = [f for f in freq.index if f in s.index]
    q = s.loc[common, "q_bh"].clip(lower=1e-300)
    x = -np.log10(q)
    y = freq.loc[common]
    sig = q < 0.05
    fig = go.Figure(go.Scatter(x=x, y=y, mode="markers", text=common,
                               marker=dict(color=np.where(sig, PALETTE[1], PALETTE[0]), opacity=.75, size=8),
                               hovertemplate="%{text}<br>−log10 q = %{x:.2f}<br>freq = %{y:.2f}<extra></extra>"))
    fig.add_vline(x=-np.log10(0.05), line_dash="dot", line_color=MUTED, annotation_text="q = 0.05")
    fig.update_layout(title=f"Predictive (selection frequency, {method_label}) vs statistically significant "
                            "(univariate BH-FDR)",
                      xaxis_title="−log10 q (univariate)", yaxis_title="selection frequency", height=430)
    return fig


def shap_beeswarm(values: np.ndarray, names: list, color_values: pd.DataFrame, n: int = 20,
                  title: str = "SHAP summary") -> go.Figure:
    imp = np.abs(values).mean(0)
    order = np.argsort(-imp)[:n][::-1]
    rng = np.random.default_rng(0)
    fig = go.Figure()
    for rank, j in enumerate(order):
        f = names[j]
        cv = color_values[f].to_numpy(float) if f in color_values else np.full(values.shape[0], np.nan)
        if np.isfinite(cv).any():
            lo, hi = np.nanpercentile(cv, [5, 95])
            c = np.clip((cv - lo) / (hi - lo if hi > lo else 1), 0, 1)
        else:
            c = None
        fig.add_trace(go.Scatter(
            x=values[:, j], y=rank + rng.uniform(-0.3, 0.3, values.shape[0]), mode="markers",
            marker=dict(size=5, color=c if c is not None else MUTED, colorscale="RdBu_r", cmin=0, cmax=1,
                        showscale=rank == len(order) - 1 and c is not None,
                        colorbar=dict(title="feature<br>value", tickvals=[0, 1], ticktext=["low", "high"])),
            name=f, showlegend=False, hovertemplate=f"{f}<br>SHAP %{{x:.3f}}<extra></extra>"))
    fig.add_vline(x=0, line_color=MUTED)
    fig.update_layout(title=title, xaxis_title="SHAP value (impact on model output)",
                      yaxis=dict(tickmode="array", tickvals=list(range(len(order))),
                                 ticktext=[names[j] for j in order]),
                      height=140 + 26 * len(order), margin=dict(l=170))
    return fig


def importance_bar(df: pd.DataFrame, value: str, err: str | None = None, n: int = 25, title: str = "",
                   xlab: str = "") -> go.Figure:
    d = df.head(n).iloc[::-1]
    fig = go.Figure(go.Bar(x=d[value], y=d["feature"], orientation="h", marker_color=PALETTE[0],
                           error_x=dict(type="data", array=d[err].fillna(0)) if err else None))
    fig.update_layout(title=title, xaxis_title=xlab, height=120 + 22 * len(d), margin=dict(l=170))
    return fig


def waterfall(contrib: pd.Series, base: float, output_space: str, title: str, n: int = 15) -> go.Figure:
    c = contrib.reindex(contrib.abs().sort_values(ascending=False).index)
    top, rest = c.iloc[:n], c.iloc[n:].sum()
    labels = list(top.index) + ([f"{len(c) - n} other features"] if len(c) > n else [])
    vals = list(top.values) + ([rest] if len(c) > n else [])
    fig = go.Figure(go.Waterfall(orientation="h", y=["baseline"] + labels[::-1] + ["model output"],
                                 x=[base] + vals[::-1] + [0],
                                 measure=["absolute"] + ["relative"] * len(vals) + ["total"],
                                 increasing=dict(marker_color=PALETTE[1]), decreasing=dict(marker_color=PALETTE[0]),
                                 totals=dict(marker_color="#52606D")))
    fig.update_layout(title=title, xaxis_title=f"model output ({output_space})", height=170 + 24 * len(labels),
                      margin=dict(l=180))
    return fig


def dependence(x, shap_vals, feature: str, color=None, color_name: str = "") -> go.Figure:
    fig = go.Figure(go.Scatter(x=x, y=shap_vals, mode="markers",
                               marker=dict(color=color if color is not None else PALETTE[0], colorscale="Viridis",
                                           showscale=color is not None, colorbar=dict(title=color_name), opacity=.75)))
    fig.add_hline(y=0, line_color=MUTED)
    fig.update_layout(title=f"SHAP dependence: {feature}", xaxis_title=f"{feature} (model input scale)",
                      yaxis_title="SHAP value", height=400)
    return fig


def pdp_plot(r: dict, feature: str) -> go.Figure:
    g = r["grid"]
    fig = go.Figure()
    if r["categorical"]:
        fig.add_trace(go.Box(x=np.repeat(g.astype(str), r["ice"].shape[0]),
                             y=r["ice"].T.ravel(), name="ICE", marker_color=MUTED, boxpoints=False))
        fig.add_trace(go.Scatter(x=g.astype(str), y=r["average"], mode="markers+lines", name="PDP (average)",
                                 marker=dict(size=10, color=PALETTE[1])))
    else:
        for i, row in enumerate(r["ice"]):
            fig.add_trace(go.Scatter(x=g, y=row, mode="lines", line=dict(color="rgba(107,119,133,0.25)", width=1),
                                     showlegend=i == 0, name="ICE (individual rows)", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=g, y=r["average"], mode="lines", name="PDP (average)",
                                 line=dict(color=PALETTE[1], width=3)))
    fig.update_layout(title=f"Partial dependence & ICE: {feature} (n = {r['n_rows']} rows)",
                      xaxis_title=f"{feature} (original scale)", yaxis_title=r["response"], height=420)
    return fig
