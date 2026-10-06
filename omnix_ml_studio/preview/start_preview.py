"""PREVIEW ONLY: the proposed 'Start' screen (upload → data summary → suggested analysis).

Not wired into app.py. Run:  streamlit run preview/start_preview.py
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

st.set_page_config(page_title="Omnix ML Studio: start", layout="wide", initial_sidebar_state="collapsed")

from ml_studio import __version__  # noqa: E402
from ml_studio.core.advisor import profile_dataset  # noqa: E402
from ml_studio.core.data_validation import load_table  # noqa: E402
from ml_studio.ui import shell as SH  # noqa: E402

st.markdown(SH.css("classification"), unsafe_allow_html=True)
CONF = {"High": ("#1F7A4D", "#E6F4EC"), "Medium": ("#9A6A00", "#FBF3DF"), "Low": ("#5E6B7A", "#EEF2F6")}
st.markdown("""<style>
.rec { border:1px solid var(--line); border-radius:14px; padding:1.1rem 1.25rem 1rem; background:#fff;
  border-left:6px solid var(--c); margin-bottom:.9rem; }
.rec .top { display:flex; align-items:center; gap:.7rem; flex-wrap:wrap; }
.rec .mod { font-size:1.45rem; font-weight:700; color:var(--c); }
.rec .badge { font-size:.78rem; font-weight:600; padding:.15rem .6rem; border-radius:999px; color:var(--bc); background:var(--bb); }
.rec .out { color:var(--ink); font-size:1rem; margin:.35rem 0 .5rem; }
.rec ul { margin:.2rem 0 .3rem 1.1rem; padding:0; } .rec li { margin:.15rem 0; color:#2C3E50; }
.rec .cau { background:#FFF7E6; border-radius:9px; padding:.5rem .7rem; margin-top:.5rem; color:#7A4F00; font-size:.9rem; }
.alt { border:1px solid var(--line); border-radius:12px; padding:.8rem .9rem; background:#fff; border-top:4px solid var(--c);
  height:100%; }
.alt .mod { font-weight:700; color:var(--c); font-size:1.05rem; } .alt p { font-size:.88rem; color:#3D4F61; margin:.3rem 0 0; }
.sumbox { background:var(--panel); border-radius:12px; padding:.9rem 1.1rem; font-size:1.02rem; color:var(--ink);
  line-height:1.55; margin-bottom:.6rem; }
.pv { display:inline-block; background:#FFF0D6; color:#8A5A00; font-weight:600; border-radius:8px; padding:.2rem .6rem;
  font-size:.8rem; }
</style>""", unsafe_allow_html=True)

left, right = st.columns([1.15, 3], vertical_alignment="center")
left.markdown(f'<div class="omx-brand"><span class="mark">Omnix™ ML Studio</span><span class="sub">v{__version__}</span></div>',
              unsafe_allow_html=True)
right.markdown('<span class="pv">Preview of the proposed Start screen: not yet part of the app</span>',
               unsafe_allow_html=True)
st.markdown('<hr class="omx-rule">', unsafe_allow_html=True)
st.markdown('<div class="omx-step-title"><span class="t">Start with your data</span></div>'
            '<div class="omx-step-lead">Upload a table. The studio reads it, summarises what it contains and suggests '
            'which analysis fits. Nothing is decided until you confirm.</div>', unsafe_allow_html=True)

main, side = SH.workspace()
P = SH.panel(side, "start")
c = SH.card(P, "src", "Dataset", "CSV, TSV or Excel. One row per sample (or per time point).")
EX = os.path.join(ROOT, "examples")
files = sorted(f for f in os.listdir(EX) if f.endswith(".csv"))
src = c.radio("Source", ["Example dataset", "Upload a file"], key="src", label_visibility="collapsed", horizontal=True)
df, name = None, None
if src == "Upload a file":
    up = c.file_uploader("File", type=["csv", "tsv", "txt", "xlsx"], label_visibility="collapsed")
    if up is not None:
        df, name = load_table(up.getvalue(), up.name), up.name
else:
    name = c.selectbox("Example", files, index=files.index(st.query_params.get("f", "clinical_plus_800_genes.csv"))
                       if st.query_params.get("f", "clinical_plus_800_genes.csv") in files else 0, key="ex")
    df = load_table(os.path.join(EX, name))

if df is None:
    SH.empty(main, "No dataset yet", "Upload a file in the panel to see the summary and the suggested analysis.")
    st.stop()

pr = profile_dataset(df)
P.markdown("&nbsp;", unsafe_allow_html=True)
P.button(f"Open in {SH.MOD[pr.suggestions[0].module].name}", type="primary", use_container_width=True, disabled=True,
         help="Preview: in the app this opens the module with the outcome and roles pre-filled.")
P.caption("In the app, this button opens the suggested module with the dataset loaded and the roles pre-filled. "
          "You can also open any other module.")

SH.section(main, f"What is in {name}")
main.markdown(f'<div class="sumbox">{pr.summary}</div>', unsafe_allow_html=True)
mf = pr.missing_fraction
_special = set(pr.id_columns + pr.group_columns + pr.date_columns + [pr.suggestions[0].outcome])
SH.kpis(main, [("time points" if pr.suggestions[0].module == "timeseries" else "samples", f"{pr.n_rows:,}"),
               ("columns", f"{pr.n_cols:,}"), ("numeric features", f"{len([c for c in pr.numeric if c not in _special]):,}"),
               ("categorical features", f"{len([c for c in pr.categorical if c not in _special]):,}"),
               ("missing cells", "none" if mf == 0 else ("< 0.1%" if mf < 0.001 else f"{mf:.1%}"))])


def outcome_line(s):
    if s.module in ("classification", "regression"):
        return f"Outcome: <b>{s.outcome}</b> ({s.task})"
    if s.module == "timeseries":
        return f"Forecast <b>{s.outcome}</b> over <b>{pr.roles['date']}</b>"
    return (f"Reference label kept aside for evaluation: <b>{s.outcome}</b>" if s.outcome
            else "No outcome column needed")


top = pr.suggestions[0]
m = SH.MOD[top.module]
bc, bb = CONF[top.confidence]
SH.section(main, "Recommended analysis")
reasons = "".join(f"<li>{r[0].upper() + r[1:]}</li>" for r in top.reasons)
cautions = "".join(f"<div class='cau'>{x}</div>" for x in top.cautions)
main.markdown(f"""<div class="rec" style="--c:{m.color}; --bc:{bc}; --bb:{bb}">
<div class="top"><span class="mod">{m.name}</span><span class="badge">{top.confidence} confidence</span></div>
<div class="out">{outcome_line(top)}</div><ul>{reasons}</ul>{cautions}</div>""", unsafe_allow_html=True)

alts = pr.suggestions[1:]
if alts:
    SH.section(main, "Other options")
    cols = main.columns(len(alts))
    for col, s in zip(cols, alts):
        mm = SH.MOD[s.module]
        bc, bb = CONF[s.confidence]
        col.markdown(f"""<div class="alt" style="--c:{mm.color}"><span class="mod">{mm.name}</span>
<span class="badge" style="font-size:.72rem;font-weight:600;padding:.1rem .5rem;border-radius:999px;color:{bc};background:{bb};margin-left:.4rem">{s.confidence}</span>
<p>{outcome_line(s)}</p><p>{s.reasons[0][0].upper() + s.reasons[0][1:]}.</p></div>""", unsafe_allow_html=True)

SH.section(main, "Detected column roles", "Suggested from names and values. You can change every role in the Data step.")
rows = []
role_of = {}
for c_ in pr.id_columns:
    role_of[c_] = "Sample ID"
for c_ in pr.group_columns:
    role_of[c_] = "Patient / group (repeated rows)"
for c_ in pr.date_columns:
    role_of[c_] = "Date / time"
if top.outcome:
    role_of[top.outcome] = ("Outcome" if top.module in ("classification", "regression") else
                            "Value to forecast" if top.module == "timeseries" else "Reference label (evaluation only)")
for col_, role in role_of.items():
    s = df[col_]
    rows.append({"column": col_, "suggested role": role, "type": "numeric" if pd.api.types.is_numeric_dtype(s) else "text",
                 "distinct values": f"{int(s.nunique()):,}", "missing": f"{s.isna().mean():.0%}"})
feat_note = []
om = [x for x in pr.omics_features if x not in role_of]
cl = [x for x in pr.clinical_features if x not in role_of]
if len(om) >= 10:
    rows.append({"column": f"{len(om):,} columns ({', '.join(om[:4])}, …)", "suggested role": "Features: genes / omics",
                 "type": "numeric", "distinct values": "", "missing": f"{df[om].isna().values.mean():.0%}"})
    rows.append({"column": f"{len(cl):,} columns ({', '.join(cl[:4])}{', …' if len(cl) > 4 else ''})",
                 "suggested role": "Features: clinical", "type": "mixed", "distinct values": "",
                 "missing": f"{df[cl].isna().values.mean():.0%}" if cl else "0%"})
else:
    fe = [x for x in df.columns if x not in role_of]
    if fe:
        rows.append({"column": f"{len(fe):,} columns ({', '.join(map(str, fe[:4]))}{', …' if len(fe) > 4 else ''})",
                     "suggested role": "Features", "type": "mixed", "distinct values": "",
                     "missing": f"{df[fe].isna().values.mean():.0%}"})
main.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

cands = [x for x in pr.candidates if x.score >= 1.5][:6]
if cands:
    with main.expander(f"Possible outcome columns considered ({len(cands)})", expanded=False):
        st.dataframe(pd.DataFrame([{"column": x.column, "would be": x.task, "values": x.levels,
                                    "smallest class": f"{x.minority_share:.0%}" if x.minority_share is not None else "",
                                    "why": "; ".join(x.reasons)} for x in cands]), hide_index=True, width="stretch")
main.caption("Suggestions come from column names, value types and counts. They are a starting point, not a decision: "
             "always confirm the outcome and the column roles yourself.")
