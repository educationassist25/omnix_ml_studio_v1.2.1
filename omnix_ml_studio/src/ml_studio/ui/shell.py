"""Application shell: design tokens, header + module bar, step rail, settings panel helpers.

Design notes
* One memorable device: each module has an identity colour, used only for its chip in the
  module bar and its step rail. Everything else stays ink-on-white and quiet.
* Steps are a real sequence, so they are numbered. Status is encoded by fill:
  done = tinted with a check, current = solid module colour, available = outline,
  locked = grey and disabled.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st

INK, MUTED, LINE, PAPER, PANEL = "#16324F", "#5E6B7A", "#DCE3EA", "#FFFFFF", "#F4F7FA"


@dataclass(frozen=True)
class Module:
    key: str
    name: str
    color: str
    tint: str
    blurb: str


MODULES = [
    Module("classification", "Classification", "#3B4CCA", "#ECEEFB", "Predict a category: binary or multiclass outcomes"),
    Module("regression", "Regression", "#0F7C80", "#E5F3F3", "Predict a continuous value"),
    Module("clustering", "Clustering", "#7A4FB5", "#F1ECF8", "Find groups of similar samples without labels"),
    Module("anomaly", "Anomaly detection", "#B23A48", "#F8EBED", "Flag samples that do not fit the rest"),
    Module("timeseries", "Time series", "#A86B12", "#F7F0E4", "Forecast future values of a series"),
]
MOD = {m.key: m for m in MODULES}


def css(current: str = "classification") -> str:
    cm = MOD.get(current, MODULES[0])
    mod_rules = []
    for m in MODULES:
        mod_rules.append(f"""
.st-key-mod_{m.key}_on button {{ background:{m.color}; border-color:{m.color}; color:#fff;
  box-shadow:0 2px 6px {m.color}40; }}
.st-key-mod_{m.key}_off button:hover {{ border-color:{m.color}; color:{m.color}; background:{m.tint}; }}
.st-key-mod_{m.key}_off button::before {{ content:""; display:inline-block; width:.6rem; height:.6rem;
  border-radius:50%; background:{m.color}; margin-right:.5rem; }}
[class*="st-key-step_{m.key}_"][class*="_current"] button {{ background:{m.color}; border-color:{m.color};
  color:#fff; box-shadow:0 3px 10px {m.color}45; }}
[class*="st-key-step_{m.key}_"][class*="_done"] button {{ background:{m.tint}; border-color:{m.color}66;
  color:{m.color}; }}
[class*="st-key-step_{m.key}_"][class*="_ready"] button {{ border-color:{m.color}; color:{m.color}; }}
[class*="st-key-step_{m.key}_"][class*="_ready"] button:hover,
[class*="st-key-step_{m.key}_"][class*="_done"] button:hover {{ background:{m.tint}; border-color:{m.color};
  color:{m.color}; }}
.omx-accent-{m.key} {{ color:{m.color}; }}
[class*="st-key-run_{m.key}"] button {{ background:{m.color}; border-color:{m.color}; color:#fff; font-weight:600;
  min-height:2.9rem; font-size:1rem; box-shadow:0 2px 6px {m.color}40; }}
[class*="st-key-run_{m.key}"] button:hover, [class*="st-key-run_{m.key}"] button:focus,
[class*="st-key-run_{m.key}"] button:active {{ background:{m.color}; border-color:{m.color}; color:#fff;
  filter:brightness(1.1); }}
[class*="st-key-run_{m.key}"] button:disabled {{ background:{m.tint}; border-color:{m.tint}; color:{m.color}99;
  filter:none; box-shadow:none; }}
""")
    c, t = cm.color, cm.tint
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
:root {{ --ink:{INK}; --muted:{MUTED}; --line:{LINE}; --panel:{PANEL}; --mod:{c}; --tint:{t}; }}
html, body, [class*="css"], .stApp, .stMarkdown, button, input, textarea, select {{
  font-family: 'IBM Plex Sans', 'Source Sans Pro', system-ui, sans-serif !important; }}
.stApp {{ background:{PAPER}; }}
header[data-testid="stHeader"] {{ background: transparent; height: 0; }}
.block-container {{ padding-top: 1.2rem; padding-bottom: 4rem; max-width: 1480px; }}
[data-testid="stSidebar"] {{ display:none; }}
h1, h2, h3, h4 {{ color: var(--ink); letter-spacing: -0.012em; }}
p, li {{ max-width: 78ch; }}

/* header */
.omx-brand {{ display:flex; align-items:baseline; gap:.7rem; margin: .2rem 0 .1rem; }}
.omx-brand .mark {{ font-size: 1.36rem; font-weight: 700; color: var(--ink); letter-spacing: -.02em; }}
.omx-brand .sub {{ color: var(--muted); font-size: .9rem; }}

/* every ordinary button: clearly a button */
[data-testid="stBaseButton-secondary"], [data-testid="stPopoverButton"] {{
  background:#fff; border:1.5px solid #AEBBC8; color:var(--ink); font-weight:600; border-radius:9px;
  min-height:2.45rem; transition: background .12s, border-color .12s, color .12s; }}
[data-testid="stBaseButton-secondary"]:hover, [data-testid="stPopoverButton"]:hover {{
  border-color:{c}; color:{c}; background:{t}; }}
[data-testid="stBaseButton-secondary"]:disabled {{ background:var(--panel); color:#9AA6B2; border-color:var(--line); }}
[data-testid="stBaseButton-primary"] {{ background:{c}; border:1.5px solid {c}; color:#fff; font-weight:600;
  border-radius:9px; min-height:2.45rem; box-shadow:0 2px 6px {c}40; }}
[data-testid="stBaseButton-primary"]:hover {{ background:{c}; border-color:{c}; color:#fff; filter:brightness(1.08); }}
[data-testid="stBaseButton-primary"]:disabled {{ background:{t}; border-color:{t}; color:{c}99; box-shadow:none; }}
[data-testid="stDownloadButton"] button {{ min-height:2.45rem; }}

/* module bar */
[class*="st-key-mod_"] button {{ border-radius: 999px; border:1.5px solid #C5D0DB; background:#fff;
  color: var(--ink); font-weight: 600; padding: .35rem .95rem; min-height: 2.6rem; width: 100%; font-size:.95rem; }}

/* step rail: number + title, with the status written underneath */
.omx-rule {{ border:0; border-top:1px solid var(--line); margin:.5rem 0 .7rem; }}
[class*="st-key-step_"] button {{ border-radius: 12px; border:1.5px solid var(--line); background:#fff;
  color: var(--ink); font-weight: 600; width:100%; min-height: 3.5rem; padding: .3rem .4rem; font-size: .93rem;
  flex-direction: column; gap: .05rem; line-height:1.2; }}
[class*="st-key-step_"] button::after {{ font-size:.72rem; font-weight:500; opacity:.85; }}
[class*="st-key-step_"][class*="_current"] button::after {{ content:"Current step"; }}
[class*="st-key-step_"][class*="_done"] button::after {{ content:"Completed"; }}
[class*="st-key-step_"][class*="_ready"] button::after {{ content:"Ready"; }}
[class*="st-key-step_"][class*="_locked"] button::after {{ content:"Locked"; }}
[class*="st-key-step_"][class*="_locked"] button {{ background: var(--panel); color:#9AA6B2; border-color: var(--panel); }}

/* step heading */
.omx-step-title {{ display:flex; align-items:baseline; gap:.6rem; margin:.1rem 0 .15rem; }}
.omx-step-title .n {{ font-weight:700; font-size:1.5rem; }}
.omx-step-title .t {{ font-weight:600; font-size:1.5rem; color:var(--ink); }}
.omx-step-lead {{ color: var(--muted); margin-bottom: .9rem; max-width: 92ch; }}

/* segmented "tabs": the selected option is filled */
[data-testid="stButtonGroup"] button {{ font-weight:600; min-height:2.3rem; border:1.5px solid #AEBBC8; color:var(--ink); background:#fff; }}
[data-testid="stButtonGroup"] button:hover {{ border-color:{c}; color:{c}; }}
[data-testid="stBaseButton-segmented_controlActive"], [data-testid="stButtonGroup"] button[aria-checked="true"] {{
  background:{c} !important; border-color:{c} !important; color:#fff !important; }}

/* option lists (plots, methods): full-width rows, selected row highlighted */
[class*="st-key-card_"] [data-testid="stRadio"] label {{ padding:.3rem .5rem; border-radius:8px; width:100%;
  margin:0 0 .1rem; }}
[class*="st-key-card_"] [data-testid="stRadio"] label:hover {{ background:{t}; }}
[class*="st-key-card_"] [data-testid="stRadio"] label:has(input:checked) {{ background:{t}; color:{c};
  box-shadow: inset 3px 0 0 {c}; }}
[class*="st-key-card_"] [data-testid="stRadio"] label:has(input:checked) p {{ font-weight:600; color:{c}; }}

/* settings panel */
.st-key-settings_panel {{ background: var(--panel); border-radius: 14px; padding: .9rem .95rem .7rem;
  border:1px solid var(--line); }}
.omx-panel-h {{ font-weight: 600; color: var(--ink); font-size: .98rem; margin: .1rem 0 .1rem; }}
.omx-panel-note {{ color: var(--muted); font-size: .82rem; margin: -.1rem 0 .4rem; }}
[class*="st-key-card_"] {{ background:#fff; border:1px solid var(--line); border-radius: 11px; padding: .75rem .85rem .35rem; }}
.omx-card-h {{ font-weight: 650; color: var(--ink); font-size: .95rem; margin-bottom: .25rem; }}
[class*="st-key-card_"] [data-testid="stCheckbox"] {{ margin-bottom:-.55rem; }}
[class*="st-key-card_"] [data-testid="stCheckbox"] label p {{ font-size:.9rem; }}
[data-testid="stExpander"] summary {{ font-weight:600; }}
[data-testid="stFileUploaderDropzone"] {{ border:1.5px dashed {c}88; background:#fff; }}

/* train / test split readout */
.omx-split {{ display:grid; grid-template-columns:1fr 1fr; gap:.4rem; margin:.1rem 0 .6rem; }}
.omx-split span {{ border-radius:9px; padding:.4rem .55rem; font-size:.78rem; line-height:1.25; }}
.omx-split b {{ display:block; font-size:1.15rem; }}
.omx-split .tr {{ background:{t}; color:{c}; }}
.omx-split .te {{ background:#F8EBED; color:#B23A48; }}

/* step complete bar */
.st-key-nextbar {{ background:{t}; border:1px solid {c}55; border-left:5px solid {c}; border-radius:11px;
  padding:.55rem .8rem; margin-bottom:.9rem; }}
.omx-next {{ color:{c}; font-weight:600; }}

/* results */
.omx-section {{ font-weight: 600; color: var(--ink); font-size: 1.1rem; margin: 1.1rem 0 .3rem; }}
.omx-kpi {{ border-left: 3px solid {c}55; padding: .15rem 0 .15rem .75rem; margin-bottom:.4rem; }}
.omx-kpi .v {{ font-size: 1.55rem; font-weight: 600; color: var(--ink); line-height: 1.15; }}
.omx-kpi .k {{ color: var(--muted); font-size: .84rem; }}
.omx-empty {{ border: 1.5px dashed var(--line); border-radius: 14px; padding: 2.2rem 1.6rem; color: var(--muted);
  text-align: left; }}
.omx-empty b {{ color: var(--ink); }}
.omx-chip {{ display:inline-block; border:1px solid var(--line); border-radius:999px; padding:.05rem .55rem;
  font-size:.78rem; color: var(--muted); margin: 0 .25rem .25rem 0; background:#fff; }}
.omx-pipe {{ display:flex; flex-wrap:wrap; align-items:center; gap:.35rem; margin:.3rem 0 .8rem; }}
.omx-pipe .s {{ border:1px solid var(--line); border-radius:8px; padding:.3rem .6rem; background:#fff; font-size:.85rem; }}
.omx-pipe .a {{ color:#9AA6B2; }}
[data-testid="stDataFrame"] {{ border-radius: 10px; }}
button:focus-visible {{ outline: 2px solid {INK} !important; outline-offset: 2px; }}
@media (max-width: 900px) {{ .omx-brand .sub {{ display:none; }} }}
@media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; animation: none !important; }} }}
{''.join(mod_rules)}
</style>
"""


# --------------------------------------------------------------------------- header & nav
def header(version: str):
    left, right = st.columns([1.15, 3], vertical_alignment="center")
    left.markdown(f'<div class="omx-brand"><span class="mark">Omnix™ ML Studio</span>'
                  f'<span class="sub">v{version}</span></div>', unsafe_allow_html=True)
    cur = st.session_state.get("module", "classification")
    cols = right.columns([1.05, 0.95, 0.95, 1.2, 1.0])
    for c, m in zip(cols, MODULES):
        if c.button(m.name, key=f"mod_{m.key}_{'on' if m.key == cur else 'off'}", help=m.blurb,
                    use_container_width=True):
            st.session_state["module"] = m.key
            st.rerun()
    return MOD[cur]


@dataclass
class Step:
    key: str
    title: str
    lead: str
    render: Callable
    unlocked: bool = True
    done: bool = False


def step_rail(module: Module, steps: list[Step]) -> Step:
    sk = f"step_{module.key}"
    cur = st.session_state.get(sk, steps[0].key)
    if cur not in {s.key for s in steps} or not next(s for s in steps if s.key == cur).unlocked:
        cur = steps[0].key
    st.markdown('<hr class="omx-rule">', unsafe_allow_html=True)
    cols = st.columns(len(steps), gap="small")
    for i, (c, s) in enumerate(zip(cols, steps), start=1):
        status = "current" if s.key == cur else ("done" if s.done else ("ready" if s.unlocked else "locked"))
        label = f"{'✓ ' if s.done and s.key != cur else ''}{i} · {s.title}"
        if c.button(label, key=f"step_{module.key}_{s.key}_{status}", disabled=not s.unlocked,
                    use_container_width=True, help=None if s.unlocked else "Complete the previous steps first"):
            st.session_state[sk] = s.key
            st.rerun()
    st.markdown('<hr class="omx-rule">', unsafe_allow_html=True)
    step = next(s for s in steps if s.key == cur)
    n = [s.key for s in steps].index(cur) + 1
    st.markdown(f'<div class="omx-step-title"><span class="n omx-accent-{module.key}">{n}</span>'
                f'<span class="t">{step.title}</span></div><div class="omx-step-lead">{step.lead}</div>',
                unsafe_allow_html=True)
    return step


def go_to(module: Module, step_key: str):
    st.session_state[f"step_{module.key}"] = step_key


def next_banner(module: Module, steps: list[Step], current: Step):
    """Shown at the top of a finished step: a clear way forward without hiding the results."""
    keys = [s.key for s in steps]
    i = keys.index(current.key)
    if not current.done or i + 1 >= len(steps) or not steps[i + 1].unlocked:
        return
    nxt = steps[i + 1]
    box = st.container(key="nextbar")
    a, b = box.columns([3, 1.1], vertical_alignment="center")
    a.markdown(f'<span class="omx-next">✓ {current.title} complete.</span> Review the results below, or continue.',
               unsafe_allow_html=True)
    if b.button(f"Continue to {nxt.title}  →", key=f"nexttop_{module.key}_{current.key}", type="primary",
                use_container_width=True):
        go_to(module, nxt.key)
        st.rerun()


def advance(module: Module, step_key: str):
    """Move to ``step_key`` after an action that completes the current step (call before st.rerun)."""
    go_to(module, step_key)


def next_button(module: Module, steps: list[Step], current: Step, label: Optional[str] = None):
    keys = [s.key for s in steps]
    i = keys.index(current.key)
    if i + 1 < len(steps) and steps[i + 1].unlocked:
        nxt = steps[i + 1]
        if st.button(label or f"Continue to {nxt.title}  →", key=f"next_{module.key}_{current.key}",
                     type="primary" if current.done else "secondary"):
            go_to(module, nxt.key)
            st.rerun()


# --------------------------------------------------------------------------- layout pieces
def workspace():
    """(main, panel) columns. The panel holds grouped settings and the Run action."""
    main, side = st.columns([2.35, 1], gap="large")
    return main, side


def panel(side, key: str):
    return side.container(key="settings_panel")


def card(parent, key: str, title: str, note: str = ""):
    c = parent.container(key=f"card_{key}")
    c.markdown(f'<div class="omx-card-h">{title}</div>' + (f'<div class="omx-panel-note">{note}</div>' if note else ""),
               unsafe_allow_html=True)
    return c


def run_button(parent, module: Module, label: str, disabled: bool = False, key_suffix: str = "") -> bool:
    return parent.button(label, key=f"run_{module.key}{key_suffix}", disabled=disabled, use_container_width=True)


def section(parent, title: str, note: str = ""):
    parent.markdown(f'<div class="omx-section">{title}</div>', unsafe_allow_html=True)
    if note:
        parent.caption(note)


def kpis(parent, items: list[tuple[str, str]]):
    cols = parent.columns(len(items))
    for c, (k, v) in zip(cols, items):
        c.markdown(f'<div class="omx-kpi"><div class="v">{v}</div><div class="k">{k}</div></div>',
                   unsafe_allow_html=True)


def empty(parent, title: str, text: str):
    parent.markdown(f'<div class="omx-empty"><b>{title}</b><br>{text}</div>', unsafe_allow_html=True)


def pipeline_strip(parent, steps: list[str]):
    html = '<span class="a">→</span>'.join(f'<span class="s">{s}</span>' for s in steps)
    parent.markdown(f'<div class="omx-pipe">{html}</div>', unsafe_allow_html=True)


def chips(parent, items: list[str]):
    parent.markdown("".join(f'<span class="omx-chip">{i}</span>' for i in items), unsafe_allow_html=True)


def picker(parent, key: str, label: str, options: list, format_func=str, default=None, help=None):
    """Single choice shown as a compact segmented control (falls back to radio)."""
    try:
        v = parent.segmented_control(label, options, format_func=format_func, default=default or options[0],
                                     key=key, help=help)
        return v if v is not None else (default or options[0])
    except AttributeError:
        return parent.radio(label, options, format_func=format_func, key=key, horizontal=True, help=help)
