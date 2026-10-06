from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from . import MUTED, PALETTE


def roc(y, proba, classes, title="ROC curve", only=None) -> go.Figure:
    """``only``: class indices to draw (multiclass one-vs-rest); default all."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color=MUTED),
                             name="chance", hoverinfo="skip"))
    if proba.shape[1] == 2:
        from sklearn.metrics import roc_auc_score
        fpr, tpr, _ = roc_curve(y, proba[:, 1])
        auc = roc_auc_score(y, proba[:, 1]) if 0 < np.sum(y) < len(y) else float("nan")
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{classes[1]} vs {classes[0]} (AUC {auc:.3f})",
                                 line=dict(width=2.5, color=PALETTE[0])))
    else:
        from sklearn.metrics import roc_auc_score
        for k, c in enumerate(classes):
            if only is not None and k not in only:
                continue
            if 0 < (y == k).sum() < len(y):
                fpr, tpr, _ = roc_curve((y == k).astype(int), proba[:, k])
                auc = roc_auc_score((y == k).astype(int), proba[:, k])
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{c} vs rest (AUC {auc:.3f})",
                                         line=dict(width=2.2, color=PALETTE[k % len(PALETTE)])))
    fig.update_layout(title=title, xaxis_title="1 − specificity", yaxis_title="Sensitivity",
                      xaxis_range=[0, 1], yaxis_range=[0, 1.02], height=420)
    return fig


def pr(y, proba, classes, title="Precision–recall curve", only=None) -> go.Figure:
    fig = go.Figure()
    if proba.shape[1] == 2:
        p, r, _ = precision_recall_curve(y, proba[:, 1])
        fig.add_trace(go.Scatter(x=r, y=p, mode="lines", name=str(classes[1]),
                                 line=dict(width=2.5, color=PALETTE[0])))
        fig.add_hline(y=float(np.mean(y == 1)), line_dash="dash", line_color=MUTED,
                      annotation_text="prevalence", annotation_position="bottom right")
    else:
        for k, c in enumerate(classes):
            if only is not None and k not in only:
                continue
            if (y == k).any():
                p, r, _ = precision_recall_curve((y == k).astype(int), proba[:, k])
                fig.add_trace(go.Scatter(x=r, y=p, mode="lines", name=f"{c} vs rest",
                                         line=dict(width=2.2, color=PALETTE[k % len(PALETTE)])))
    fig.update_layout(title=title, xaxis_title="Recall", yaxis_title="Precision",
                      xaxis_range=[0, 1], yaxis_range=[0, 1.02], height=420)
    return fig


def confusion(y, y_pred, classes, title="Confusion matrix", normalize="row") -> go.Figure:
    """``normalize``: 'row' (recall per observed class), 'column' (precision per predicted class) or 'none'."""
    k = len(classes)
    cm = confusion_matrix(y, y_pred, labels=list(range(k)))
    if normalize == "column":
        pct = cm / np.maximum(cm.sum(0, keepdims=True), 1)
    elif normalize == "none":
        pct = cm / max(cm.max(), 1)
    else:
        pct = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    text = [[f"{cm[i, j]}" + ("" if normalize == "none" else f"<br>{pct[i, j]:.0%}") for j in range(k)] for i in range(k)]
    fig = go.Figure(go.Heatmap(z=pct, x=[str(c) for c in classes], y=[str(c) for c in classes],
                               text=text, texttemplate="%{text}", colorscale="Blues", zmin=0, zmax=1,
                               colorbar=dict(title={"row": "row %", "column": "column %", "none": "relative"}[normalize])))
    fig.update_layout(title=title, xaxis_title="Predicted", yaxis_title="Observed",
                      yaxis_autorange="reversed", height=380 + 20 * k)
    fig.update_xaxes(type="category")
    fig.update_yaxes(type="category")
    return fig


def calibration(y, proba, n_bins=10, title="Calibration") -> go.Figure:
    p = proba[:, 1]
    bins = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    bins[0], bins[-1] = 0, 1
    idx = np.clip(np.digitize(p, bins[1:-1]), 0, n_bins - 1)
    mp = [p[idx == b].mean() for b in range(n_bins) if (idx == b).any()]
    ob = [y[idx == b].mean() for b in range(n_bins) if (idx == b).any()]
    fig = go.Figure([go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color=MUTED),
                                name="perfect"),
                     go.Scatter(x=mp, y=ob, mode="lines+markers", name="model",
                                line=dict(color=PALETTE[0], width=2.5))])
    fig.update_layout(title=title, xaxis_title="Mean predicted probability",
                      yaxis_title="Observed fraction", height=380)
    return fig


def threshold_analysis(y, proba, title="Threshold analysis") -> go.Figure:
    """Precision, recall, F1, specificity and predicted-positive rate across decision thresholds."""
    from sklearn.metrics import f1_score, precision_score, recall_score
    p = proba[:, 1]
    ts = np.linspace(0.02, 0.98, 49)
    rows = {"precision": [], "recall (sensitivity)": [], "specificity": [], "F1": [], "predicted positive": []}
    for t in ts:
        yp = (p >= t).astype(int)
        rows["precision"].append(precision_score(y, yp, zero_division=0))
        rows["recall (sensitivity)"].append(recall_score(y, yp, zero_division=0))
        rows["specificity"].append(((yp == 0) & (y == 0)).sum() / max((y == 0).sum(), 1))
        rows["F1"].append(f1_score(y, yp, zero_division=0))
        rows["predicted positive"].append(yp.mean())
    fig = go.Figure()
    for i, (k, v) in enumerate(rows.items()):
        fig.add_trace(go.Scatter(x=ts, y=v, mode="lines", name=k,
                                 line=dict(width=3 if k == "F1" else 2, dash="dot" if k == "predicted positive" else None,
                                           color=PALETTE[i % len(PALETTE)])))
    best = ts[int(np.argmax(rows["F1"]))]
    fig.add_vline(x=best, line_dash="dash", line_color=MUTED, annotation_text=f"max F1 at {best:.2f}")
    fig.update_layout(title=title, xaxis_title="decision threshold", yaxis_title="value", yaxis_range=[0, 1.02],
                      height=430)
    return fig


def gain_lift(y, proba, title="Cumulative gain and lift") -> go.Figure:
    from plotly.subplots import make_subplots
    p = proba[:, 1]
    order = np.argsort(-p)
    yy = np.asarray(y)[order]
    frac = np.arange(1, len(yy) + 1) / len(yy)
    gain = np.cumsum(yy) / max(yy.sum(), 1)
    lift = gain / frac
    fig = make_subplots(1, 2, subplot_titles=("Cumulative gain", "Lift"))
    fig.add_trace(go.Scatter(x=frac, y=gain, mode="lines", name="model", line=dict(color=PALETTE[0], width=2.5)), 1, 1)
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="random", line=dict(dash="dash", color=MUTED)), 1, 1)
    fig.add_trace(go.Scatter(x=frac, y=lift, mode="lines", name="lift", line=dict(color=PALETTE[2], width=2.5),
                             showlegend=False), 1, 2)
    fig.add_hline(y=1, line_dash="dash", line_color=MUTED, row=1, col=2)
    fig.update_xaxes(title_text="fraction of samples (ranked by score)", tickformat=".0%")
    fig.update_layout(title=title, height=400)
    return fig


def class_report(y, y_pred, classes, title="Per-class report") -> go.Figure:
    from sklearn.metrics import precision_recall_fscore_support
    p, r, f, s = precision_recall_fscore_support(y, y_pred, labels=list(range(len(classes))), zero_division=0)
    z = np.c_[p, r, f]
    fig = go.Figure(go.Heatmap(z=z, x=["precision", "recall", "F1"], y=[f"{c} (n={n})" for c, n in zip(classes, s)],
                               zmin=0, zmax=1, colorscale="Blues", text=np.round(z, 2), texttemplate="%{text}"))
    fig.update_layout(title=title, height=180 + 50 * len(classes), yaxis_autorange="reversed")
    fig.update_yaxes(type="category")
    return fig


def learning_curve_fig(sizes, train_scores, val_scores, metric_label, title="Learning curve") -> go.Figure:
    fig = go.Figure()
    for name, sc, col in (("training", train_scores, PALETTE[1]), ("cross-validation", val_scores, PALETTE[0])):
        m, s = sc.mean(1), sc.std(1)
        fig.add_trace(go.Scatter(x=np.r_[sizes, sizes[::-1]], y=np.r_[m + s, (m - s)[::-1]], fill="toself",
                                 fillcolor=col, opacity=.15, line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=sizes, y=m, mode="lines+markers", name=name, line=dict(color=col, width=2.5)))
    fig.update_layout(title=title, xaxis_title="training samples", yaxis_title=metric_label, height=420)
    return fig
