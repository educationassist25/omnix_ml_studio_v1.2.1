from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..core.metrics import LOWER_IS_BETTER, METRIC_LABELS
from . import MUTED, PALETTE


def leaderboard_bars(lb: pd.DataFrame, metric: str) -> go.Figure:
    d = lb.sort_values(f"{metric}_mean", ascending=metric not in LOWER_IS_BETTER)
    col = [MUTED if k == "dummy" else PALETTE[0] for k in d["key"]]
    fig = go.Figure(go.Bar(x=d[f"{metric}_mean"], y=d["Model"], orientation="h", marker_color=col,
                           error_x=dict(type="data", array=d[f"{metric}_sd"].fillna(0), color="#52606D"),
                           hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    fig.update_layout(title=f"Cross-validated {METRIC_LABELS.get(metric, metric)} (mean ± SD across folds)",
                      xaxis_title=METRIC_LABELS.get(metric, metric), height=110 + 34 * len(d),
                      margin=dict(l=220))
    return fig


def fold_strip(folds: pd.DataFrame, lb: pd.DataFrame, metric: str) -> go.Figure:
    d = folds[folds["metric"] == metric]
    names = dict(zip(lb["key"], lb["Model"]))
    order = list(lb["key"])
    fig = go.Figure()
    for i, k in enumerate(order):
        v = d.loc[d["key"] == k, "value"]
        fig.add_trace(go.Box(x=v, name=names[k], boxpoints="all", jitter=.4, pointpos=0,
                             marker_color=MUTED if k == "dummy" else PALETTE[i % len(PALETTE)],
                             showlegend=False))
    fig.update_layout(title=f"Per-fold {METRIC_LABELS.get(metric, metric)}", height=110 + 34 * len(order),
                      margin=dict(l=220), xaxis_title=METRIC_LABELS.get(metric, metric))
    return fig


def ci_forest(ci: pd.DataFrame, title="Holdout metrics with 95% bootstrap CI") -> go.Figure:
    d = ci.dropna(subset=["estimate"]).iloc[::-1]
    fig = go.Figure(go.Scatter(
        x=d["estimate"], y=d["label"], mode="markers", marker=dict(size=10, color=PALETTE[0]),
        error_x=dict(type="data", symmetric=False, array=(d["ci_high"] - d["estimate"]).fillna(0),
                     arrayminus=(d["estimate"] - d["ci_low"]).fillna(0), color="#52606D"),
        hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    fig.update_layout(title=title, height=90 + 32 * len(d), margin=dict(l=180))
    return fig


def nested_outer(outer: pd.DataFrame, metric: str) -> go.Figure:
    col = f"outer_{metric}"
    fig = go.Figure(go.Bar(x=[f"Fold {k}" for k in outer["outer_fold"]], y=outer[col],
                           text=outer["selected_model"], textposition="outside",
                           marker_color=PALETTE[0]))
    fig.add_hline(y=float(outer[col].mean()), line_dash="dash", line_color=PALETTE[1],
                  annotation_text=f"mean {outer[col].mean():.3f}")
    fig.update_layout(title=f"Outer-fold {METRIC_LABELS.get(metric, metric)} (label = model chosen by inner CV)",
                      yaxis_title=METRIC_LABELS.get(metric, metric), height=380)
    return fig


def shift_bars(tab: pd.DataFrame, n=20) -> go.Figure:
    d = tab.head(n).iloc[::-1]
    fig = go.Figure(go.Bar(x=d["smd"], y=d["feature"], orientation="h",
                           marker_color=np.where(d["abs_smd"] > 0.5, PALETTE[1], PALETTE[0])))
    fig.add_vline(x=0.5, line_dash="dot", line_color=MUTED); fig.add_vline(x=-0.5, line_dash="dot", line_color=MUTED)
    fig.update_layout(title="Largest feature shifts (standardised mean difference, external − development)",
                      xaxis_title="SMD", height=120 + 22 * len(d), margin=dict(l=160))
    return fig
