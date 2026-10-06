from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import MUTED, PALETTE


def observed_vs_predicted(y, p, title="Observed vs predicted") -> go.Figure:
    lo, hi = float(min(np.min(y), np.min(p))), float(max(np.max(y), np.max(p)))
    fig = go.Figure([go.Scatter(x=p, y=y, mode="markers", marker=dict(color=PALETTE[0], opacity=.65),
                                name="samples"),
                     go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(dash="dash", color=MUTED),
                                name="identity")])
    fig.update_layout(title=title, xaxis_title="Predicted", yaxis_title="Observed", height=420)
    return fig


def residuals(y, p, title="Residuals") -> go.Figure:
    r = np.asarray(y) - np.asarray(p)
    fig = make_subplots(1, 2, subplot_titles=("Residual vs predicted", "Residual distribution"))
    fig.add_trace(go.Scatter(x=p, y=r, mode="markers", marker=dict(color=PALETTE[0], opacity=.6),
                             showlegend=False), 1, 1)
    fig.add_hline(y=0, line_dash="dash", line_color=MUTED, row=1, col=1)
    fig.add_trace(go.Histogram(x=r, marker_color=PALETTE[0], showlegend=False, nbinsx=30), 1, 2)
    fig.update_xaxes(title_text="Predicted", row=1, col=1)
    fig.update_yaxes(title_text="Observed − predicted", row=1, col=1)
    fig.update_xaxes(title_text="Residual", row=1, col=2)
    fig.update_layout(title=title, height=380)
    return fig


def qq(y, p, title="Normal Q–Q plot of residuals") -> go.Figure:
    from scipy import stats
    r = np.sort(np.asarray(y) - np.asarray(p))
    z = (r - r.mean()) / (r.std(ddof=1) or 1)
    q = stats.norm.ppf((np.arange(1, len(z) + 1) - 0.5) / len(z))
    fig = go.Figure([go.Scatter(x=q, y=z, mode="markers", marker=dict(color=PALETTE[0], opacity=.7), name="residuals"),
                     go.Scatter(x=[q.min(), q.max()], y=[q.min(), q.max()], mode="lines",
                                line=dict(dash="dash", color=MUTED), name="normal")])
    fig.update_layout(title=title, xaxis_title="theoretical quantile", yaxis_title="standardised residual", height=400)
    return fig
