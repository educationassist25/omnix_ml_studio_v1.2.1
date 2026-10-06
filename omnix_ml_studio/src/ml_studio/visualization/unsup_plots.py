"""Figures for clustering, anomaly detection and time series."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import MUTED, PALETTE

NOISE = "#B8C0C8"


def _lab(c):
    return "noise" if c == -1 else f"cluster {c}"


# --------------------------------------------------------------------------- clustering
def embedding(E, labels, title, ev=None, hover=None, color_name="cluster") -> go.Figure:
    fig = go.Figure()
    lab = np.asarray(labels)
    cats = sorted(set(lab.tolist()), key=lambda v: (v == -1, str(v)))
    for i, c in enumerate(cats):
        m = lab == c
        fig.add_trace(go.Scattergl(x=E[m, 0], y=E[m, 1], mode="markers", name=_lab(c) if color_name == "cluster" else str(c),
                                   marker=dict(size=6, opacity=.75, color=NOISE if c == -1 else PALETTE[i % len(PALETTE)]),
                                   text=None if hover is None else np.asarray(hover)[m],
                                   hovertemplate="%{text}<extra></extra>" if hover is not None else None))
    xl = f"PC1 ({ev[0]:.0%})" if ev is not None else "dimension 1"
    yl = f"PC2 ({ev[1]:.0%})" if ev is not None else "dimension 2"
    fig.update_layout(title=title, xaxis_title=xl, yaxis_title=yl, height=480, legend_title=color_name)
    return fig


def cluster_sizes(labels) -> go.Figure:
    s = pd.Series(labels).value_counts().sort_index()
    fig = go.Figure(go.Bar(x=[_lab(c) for c in s.index], y=s.values,
                           marker_color=[NOISE if c == -1 else PALETTE[i % len(PALETTE)] for i, c in enumerate(s.index)],
                           text=s.values, textposition="outside"))
    fig.update_layout(title="Cluster sizes", yaxis_title="samples", height=340)
    return fig


def silhouette_plot(Z, labels) -> go.Figure:
    from sklearn.metrics import silhouette_samples
    lab = np.asarray(labels)
    keep = lab != -1
    s = silhouette_samples(Z.to_numpy(float)[keep], lab[keep])
    fig = go.Figure()
    y0 = 0
    for i, c in enumerate(sorted(set(lab[keep]))):
        v = np.sort(s[lab[keep] == c])
        fig.add_trace(go.Bar(x=v, y=np.arange(y0, y0 + len(v)), orientation="h", name=_lab(c),
                             marker_color=PALETTE[i % len(PALETTE)], marker_line_width=0, width=1))
        y0 += len(v) + 8
    fig.add_vline(x=float(s.mean()), line_dash="dash", line_color=MUTED, annotation_text=f"mean {s.mean():.2f}")
    fig.update_layout(title="Silhouette per sample (higher = better matched to its own cluster)",
                      xaxis_title="silhouette", yaxis=dict(showticklabels=False), bargap=0, height=460)
    return fig


def k_scan_plot(scan: pd.DataFrame) -> go.Figure:
    cols = [c for c in ("silhouette", "calinski_harabasz", "davies_bouldin", "inertia", "bic") if c in scan and
            scan[c].notna().any()]
    names = {"silhouette": "Silhouette ↑", "calinski_harabasz": "Calinski–Harabasz ↑", "davies_bouldin": "Davies–Bouldin ↓",
             "inertia": "Inertia (elbow)", "bic": "BIC ↓"}
    fig = make_subplots(1, len(cols), subplot_titles=[names[c] for c in cols])
    for i, c in enumerate(cols, start=1):
        best = {"silhouette": scan[c].idxmax(), "calinski_harabasz": scan[c].idxmax(), "davies_bouldin": scan[c].idxmin(),
                "bic": scan[c].idxmin()}.get(c)
        fig.add_trace(go.Scatter(x=scan["k"], y=scan[c], mode="lines+markers", line=dict(color=PALETTE[0], width=2),
                                 showlegend=False), 1, i)
        if best is not None:
            fig.add_trace(go.Scatter(x=[scan.loc[best, "k"]], y=[scan.loc[best, c]], mode="markers",
                                     marker=dict(size=13, color=PALETTE[1], symbol="star"), showlegend=False), 1, i)
    fig.update_xaxes(title_text="k", dtick=1)
    fig.update_layout(height=330, title="Number of clusters: criteria across k (star = best by that criterion)")
    return fig


def jaccard_bars(per: pd.DataFrame) -> go.Figure:
    col = {"stable": "#4E937A", "pattern": "#D9A441", "weak": "#C0504D", "dissolved": "#8B2F2F"}
    fig = go.Figure(go.Bar(x=[f"cluster {c} (n={n})" for c, n in zip(per["cluster"], per["size"])], y=per["mean_jaccard"],
                           error_y=dict(type="data", array=per["sd_jaccard"].fillna(0)),
                           marker_color=[col[a] for a in per["assessment"]], text=per["assessment"],
                           textposition="inside", insidetextanchor="start", textfont=dict(color="white")))
    for y, t in ((0.75, "stable"), (0.5, "dissolved below")):
        fig.add_hline(y=y, line_dash="dot", line_color=MUTED, annotation_text=t, annotation_position="right")
    fig.update_layout(title="Clusterwise bootstrap stability (mean Jaccard)", yaxis_range=[0, 1.1],
                      yaxis_title="Jaccard similarity", height=380)
    return fig


def profile_heatmap(X: pd.DataFrame, labels, features: list) -> go.Figure:
    lab = np.asarray(labels)
    keep = lab != -1
    V = X.loc[keep, features].astype(float)
    z = (V - V.mean()) / V.std(ddof=0).replace(0, 1)
    m = z.groupby(lab[keep]).mean()
    fig = go.Figure(go.Heatmap(z=m.values, x=features, y=[_lab(c) for c in m.index], colorscale="RdBu_r", zmid=0,
                               colorbar=dict(title="mean z")))
    fig.update_layout(title="Cluster profiles (mean standardised value)", height=180 + 40 * len(m),
                      xaxis_tickangle=-45)
    return fig


def contingency(labels, reference) -> go.Figure:
    ct = pd.crosstab(pd.Series(reference, name="reference").astype(str), pd.Series(labels, name="cluster"))
    fig = go.Figure(go.Heatmap(z=ct.values, x=[_lab(c) for c in ct.columns], y=ct.index.astype(str), colorscale="Blues",
                               text=ct.values, texttemplate="%{text}"))
    fig.update_layout(title="Clusters versus the reference label", xaxis_title="cluster", yaxis_title="reference",
                      height=200 + 40 * len(ct))
    return fig


# --------------------------------------------------------------------------- anomaly
def score_hist(scores, flags, title="Anomaly score distribution", reference=None) -> go.Figure:
    s, f = np.asarray(scores), np.asarray(flags)
    fig = go.Figure()
    if reference is not None:
        r = np.asarray(reference).astype(int)
        fig.add_trace(go.Histogram(x=s[r == 0], name="reference: normal", marker_color=PALETTE[0], opacity=.6, nbinsx=50))
        fig.add_trace(go.Histogram(x=s[r == 1], name="reference: anomaly", marker_color=PALETTE[1], opacity=.8, nbinsx=50))
        fig.update_layout(barmode="overlay")
    else:
        fig.add_trace(go.Histogram(x=s, marker_color=PALETTE[0], nbinsx=50, name="all samples"))
    if f.any():
        thr = float(np.min(s[f == 1]))
        fig.add_vline(x=thr, line_dash="dash", line_color=PALETTE[1], annotation_text="threshold")
    fig.update_layout(title=title, xaxis_title="score (higher = more anomalous)", yaxis_title="samples", height=380)
    return fig


def agreement_heatmap(M: pd.DataFrame, names: dict, title: str) -> go.Figure:
    lab = [names.get(k, k) for k in M.columns]
    fig = go.Figure(go.Heatmap(z=M.values.astype(float), x=lab, y=lab, zmin=0, zmax=1, colorscale="Blues",
                               text=np.round(M.values.astype(float), 2), texttemplate="%{text}"))
    fig.update_layout(title=title, height=160 + 34 * len(lab), xaxis_tickangle=-35, yaxis_autorange="reversed")
    return fig


def ranked_scores(scores, flags) -> go.Figure:
    s = np.sort(np.asarray(scores))[::-1]
    n = int(np.asarray(flags).sum())
    fig = go.Figure(go.Scatter(x=np.arange(1, len(s) + 1), y=s, mode="lines", line=dict(color=PALETTE[0], width=2)))
    if n:
        fig.add_vline(x=n + .5, line_dash="dash", line_color=PALETTE[1], annotation_text=f"{n} flagged")
    fig.update_layout(title="Scores ranked (a clear elbow suggests a natural threshold)", xaxis_title="rank",
                      yaxis_title="score", height=340, xaxis_type="log")
    return fig


def deviation_bars(prof: pd.DataFrame, n: int = 15, title: str = "") -> go.Figure:
    d = prof.head(n).iloc[::-1]
    fig = go.Figure(go.Bar(x=d["robust_z"], y=d["feature"], orientation="h",
                           marker_color=[PALETTE[1] if abs(v) > 3 else PALETTE[0] for v in d["robust_z"]],
                           customdata=np.c_[d["value"], d["median"]],
                           hovertemplate="%{y}: z=%{x:.2f}<br>value %{customdata[0]:.3g}, median %{customdata[1]:.3g}<extra></extra>"))
    for x in (-3, 3):
        fig.add_vline(x=x, line_dash="dot", line_color=MUTED)
    fig.update_layout(title=title or "Most deviating features (robust z: median / MAD)", xaxis_title="robust z",
                      height=120 + 26 * len(d), margin=dict(l=160))
    return fig


# --------------------------------------------------------------------------- time series
def series_plot(y: pd.Series, title="Series", split_at=None, fh=None) -> go.Figure:
    x = y.index.astype(str)
    fig = go.Figure(go.Scatter(x=x, y=y.values, mode="lines", line=dict(color=PALETTE[0], width=2), name=y.name))
    if split_at is not None:
        fig.add_vrect(x0=x[split_at], x1=x[-1], fillcolor="#B23A48", opacity=.08, line_width=0,
                      annotation_text="sealed holdout", annotation_position="top left")
    fig.update_layout(title=title, height=360, xaxis_title=None, yaxis_title=y.name)
    return fig


def cv_origins(n: int, folds: list, n_dev: int, fh: int, index) -> go.Figure:
    fig = go.Figure()
    x = list(index.astype(str))
    for i, (a, b) in enumerate(folds):
        yv = len(folds) - i
        fig.add_trace(go.Scatter(x=[x[0], x[a - 1]], y=[yv, yv], mode="lines", line=dict(color="#C9D3DD", width=12),
                                 showlegend=i == 0, name="training window"))
        fig.add_trace(go.Scatter(x=[x[a], x[b - 1]], y=[yv, yv], mode="lines", line=dict(color=PALETTE[0], width=12),
                                 showlegend=i == 0, name="validation window"))
    if n_dev < n:
        fig.add_trace(go.Scatter(x=[x[n_dev], x[-1]], y=[0, 0], mode="lines", line=dict(color="#B23A48", width=12),
                                 name="sealed holdout"))
    fig.update_yaxes(tickvals=list(range(0, len(folds) + 1)),
                     ticktext=(["holdout"] if n_dev < n else [""]) + [f"origin {len(folds) - i}" for i in range(len(folds))][::-1])
    fig.update_layout(title=f"Expanding-window cross-validation: {len(folds)} origins, horizon {fh}", height=150 + 36 * len(folds),
                      legend=dict(orientation="h", y=-0.25))
    return fig


def forecast_plot(y: pd.Series, table: pd.DataFrame, title: str, history: int | None = None) -> go.Figure:
    hx = y.index.astype(str)
    h = slice(-history, None) if history else slice(None)
    fig = go.Figure(go.Scatter(x=hx[h], y=y.values[h], mode="lines", name="observed", line=dict(color=INK_LINE, width=2)))
    if "lower" in table:
        fig.add_trace(go.Scatter(x=list(table["time"]) + list(table["time"])[::-1],
                                 y=list(table["upper"]) + list(table["lower"])[::-1], fill="toself",
                                 fillcolor="rgba(168,107,18,.18)", line=dict(width=0), name="prediction interval",
                                 hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=table["time"], y=table["forecast"], mode="lines+markers", name="forecast",
                             line=dict(color="#A86B12", width=2.5)))
    if "actual" in table:
        fig.add_trace(go.Scatter(x=table["time"], y=table["actual"], mode="markers", name="actual (holdout)",
                                 marker=dict(color=PALETTE[1], size=7)))
    fig.update_layout(title=title, height=420)
    return fig


INK_LINE = "#2F5D8A"


def cv_forecasts(y: pd.Series, preds: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Scatter(x=y.index.astype(str), y=y.values, mode="lines", name="observed",
                               line=dict(color="#9AA6B2", width=1.5)))
    for i, (f, d) in enumerate(preds.groupby("fold")):
        fig.add_trace(go.Scatter(x=d["time"], y=d["forecast"], mode="lines+markers", name=f"origin {f}",
                                 line=dict(color=PALETTE[i % len(PALETTE)], width=2.5)))
    fig.update_layout(title="Out-of-sample forecasts at each CV origin", height=400)
    return fig


def stl_plot(stl: pd.DataFrame, y: pd.Series) -> go.Figure:
    fig = make_subplots(4, 1, shared_xaxes=True, subplot_titles=("observed", "trend", "seasonal", "remainder"),
                        vertical_spacing=.05)
    for i, (_name, v) in enumerate([("observed", y.values), ("trend", stl["trend"]), ("seasonal", stl["seasonal"]),
                                   ("remainder", stl["resid"])], start=1):
        fig.add_trace(go.Scatter(x=stl.index, y=v, mode="lines", line=dict(color=PALETTE[0] if i < 4 else MUTED, width=1.6),
                                 showlegend=False), i, 1)
    fig.update_layout(title="STL decomposition", height=640)
    return fig


def acf_plot(acf, pacf, n: int) -> go.Figure:
    fig = make_subplots(1, 2, subplot_titles=("Autocorrelation (ACF)", "Partial autocorrelation (PACF)"))
    band = 1.96 / np.sqrt(n)
    for j, v in enumerate((acf, pacf), start=1):
        lags = np.arange(len(v))
        fig.add_trace(go.Bar(x=lags[1:], y=v[1:], marker_color=PALETTE[0], showlegend=False, width=.4), 1, j)
        fig.add_hrect(y0=-band, y1=band, fillcolor="#9AA6B2", opacity=.2, line_width=0, row=1, col=j)
    fig.update_xaxes(title_text="lag")
    fig.update_layout(height=340)
    return fig


def residual_panel(resid: np.ndarray, title="Forecast residuals") -> go.Figure:
    fig = make_subplots(1, 2, subplot_titles=("over time", "distribution"))
    fig.add_trace(go.Scatter(y=resid, mode="lines+markers", line=dict(color=PALETTE[0]), showlegend=False), 1, 1)
    fig.add_hline(y=0, line_dash="dash", line_color=MUTED, row=1, col=1)
    fig.add_trace(go.Histogram(x=resid, marker_color=PALETTE[0], showlegend=False, nbinsx=25), 1, 2)
    fig.update_layout(title=title, height=340)
    return fig
