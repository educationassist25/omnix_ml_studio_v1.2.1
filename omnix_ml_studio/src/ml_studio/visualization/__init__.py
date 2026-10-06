"""Plotly figure builders (no Streamlit dependency)."""
import plotly.graph_objects as go
import plotly.io as pio

PALETTE = ["#2F5D8A", "#C0504D", "#4E937A", "#D9A441", "#7A5C99", "#5B8DB8", "#9C6B4E", "#6C757D"]
INK, MUTED, GRID = "#1F2933", "#6B7785", "#E4E7EB"

pio.templates["omnix"] = go.layout.Template(layout=dict(
    font=dict(family="Inter, Source Sans Pro, Arial, sans-serif", size=13, color=INK),
    colorway=PALETTE, paper_bgcolor="white", plot_bgcolor="white",
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor="#9AA5B1", ticks="outside"),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor="#9AA5B1", ticks="outside"),
    legend=dict(bgcolor="rgba(255,255,255,0)"),
    margin=dict(l=60, r=20, t=50, b=50), title=dict(font=dict(size=15)),
))
pio.templates.default = "omnix"
