"""Exploratory plots. These never feed back into modelling."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from . import PALETTE


def target_distribution(y: pd.Series, task: str) -> go.Figure:
    if task == "regression":
        fig = px.histogram(y.dropna(), nbins=30, color_discrete_sequence=[PALETTE[0]])
        fig.update_layout(showlegend=False, xaxis_title=y.name, yaxis_title="count")
    else:
        vc = y.astype(str).value_counts()
        fig = go.Figure(go.Bar(x=vc.index, y=vc.values, marker_color=PALETTE[0],
                               text=[f"{v} ({v / vc.sum():.0%})" for v in vc.values], textposition="outside"))
        fig.update_layout(xaxis_title=y.name, yaxis_title="count")
    fig.update_layout(title="Target distribution", height=340)
    return fig


def missingness_heatmap(X: pd.DataFrame, max_features=200, max_samples=400) -> go.Figure:
    M = X.isna()
    cols = M.mean().sort_values(ascending=False).index[:max_features]
    rows = M.index[:max_samples]
    fig = go.Figure(go.Heatmap(z=M.loc[rows, cols].astype(int).values.T, x=[str(r) for r in rows],
                               y=list(cols), colorscale=[[0, "#F5F7FA"], [1, PALETTE[1]]], showscale=False))
    fig.update_layout(title=f"Missing values (red) — top {len(cols)} features by missingness"
                      + (f", first {len(rows)} samples" if len(X) > max_samples else ""),
                      xaxis_showticklabels=False, xaxis_title="samples",
                      height=min(900, 160 + 6 * len(cols)))
    return fig


def missing_by_group(X: pd.DataFrame, g: pd.Series) -> go.Figure:
    d = X.isna().mean(axis=1).groupby(g.astype(str)).mean().sort_values()
    fig = go.Figure(go.Bar(x=d.index, y=d.values, marker_color=PALETTE[0]))
    fig.update_layout(title=f"Mean fraction missing per sample, by {g.name}", yaxis_tickformat=".0%",
                      height=320)
    return fig


def pca_scatter(scores, ev, color: pd.Series, title_note: str, hover=None) -> go.Figure:
    df = pd.DataFrame({"PC1": scores[:, 0], "PC2": scores[:, 1], color.name or "group": color.astype(str).values})
    if hover is not None:
        df["sample"] = hover
    fig = px.scatter(df, x="PC1", y="PC2", color=color.name or "group", hover_data=["sample"] if hover is not None else None,
                     color_discrete_sequence=PALETTE, opacity=.8)
    fig.update_layout(title=f"PCA (exploratory) — {title_note}",
                      xaxis_title=f"PC1 ({ev[0]:.1%})", yaxis_title=f"PC2 ({ev[1]:.1%})", height=460)
    return fig


def scree(ev) -> go.Figure:
    fig = go.Figure(go.Bar(x=[f"PC{i + 1}" for i in range(len(ev))], y=ev, marker_color=PALETTE[0]))
    fig.update_layout(title="Variance explained", yaxis_tickformat=".0%", height=300)
    return fig


def correlation_heatmap(X: pd.DataFrame, max_features=60) -> go.Figure:
    cols = X.var().sort_values(ascending=False).index[:max_features]
    C = X[cols].corr()
    fig = go.Figure(go.Heatmap(z=C.values, x=list(cols), y=list(cols), colorscale="RdBu_r", zmin=-1, zmax=1))
    fig.update_layout(title=f"Pearson correlation — {len(cols)} highest-variance features",
                      height=min(900, 200 + 12 * len(cols)), yaxis_autorange="reversed")
    return fig


def feature_box(df: pd.DataFrame, feature: str, by: str) -> go.Figure:
    fig = px.box(df, x=df[by].astype(str), y=feature, points="all", color=df[by].astype(str),
                 color_discrete_sequence=PALETTE)
    fig.update_layout(title=f"{feature} by {by}", xaxis_title=by, showlegend=False, height=380)
    return fig


def variance_summary(X: pd.DataFrame) -> go.Figure:
    v = np.log10(X.var().replace(0, np.nan).dropna())
    fig = px.histogram(v, nbins=50, color_discrete_sequence=[PALETTE[0]])
    fig.update_layout(title="Feature variance (log10)", xaxis_title="log10 variance", showlegend=False,
                      height=300)
    return fig


def outlier_scores(scores, labels, threshold=3.0) -> tuple[go.Figure, np.ndarray]:
    """Robust Mahalanobis-like distance in the first PCs (exploratory)."""
    k = min(5, scores.shape[1])
    Z = scores[:, :k]
    med = np.median(Z, 0); mad = np.median(np.abs(Z - med), 0) * 1.4826
    d = np.sqrt((((Z - med) / np.where(mad == 0, 1, mad)) ** 2).sum(1) / k)
    flag = d > threshold
    fig = go.Figure(go.Scatter(x=np.arange(len(d)), y=d, mode="markers", text=labels,
                               marker=dict(color=np.where(flag, PALETTE[1], PALETTE[0]))))
    fig.add_hline(y=threshold, line_dash="dash")
    fig.update_layout(title=f"Robust distance in first {k} PCs (flag > {threshold})", xaxis_title="sample",
                      yaxis_title="distance", height=320)
    return fig, flag
