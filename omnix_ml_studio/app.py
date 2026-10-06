"""Omnix™ ML Studio — Streamlit entry point.

Run:  streamlit run app.py
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

st.set_page_config(page_title="Omnix ML Studio", page_icon="🧬", layout="wide", initial_sidebar_state="collapsed")

from ml_studio import __version__  # noqa: E402
from ml_studio.ui import shell as SH  # noqa: E402
from ml_studio.ui import supervised  # noqa: E402
from ml_studio.utils.session_utils import get_state  # noqa: E402

st.markdown(SH.css(st.session_state.get("module", "classification")), unsafe_allow_html=True)
module = SH.header(__version__)
S = get_state(st, module.key)

from ml_studio.ui import jobs  # noqa: E402

if jobs.running():
    @st.fragment(run_every=2.0)
    def _running_jobs():
        live = jobs.running()
        if not live:
            st.rerun()
        st.caption("Running in the background: " + " · ".join(
            f"{j.label} ({j.done}/{j.total or '…'}, {int(j.elapsed // 60)} min {int(j.elapsed % 60):02d} s)" for j in live))
    _running_jobs()

try:
    if module.key in ("classification", "regression"):
        supervised.render(S, module)
    else:
        from ml_studio.ui import unsupervised
        unsupervised.render(S, module)
except MemoryError:
    st.error("The operation ran out of memory. Reduce the number of features, models or parallel workers.")

st.markdown('<hr class="omx-rule">', unsafe_allow_html=True)
with st.expander("Session log"):
    if S.history:
        st.dataframe(pd.DataFrame(S.history), hide_index=True, width="stretch")
    else:
        st.caption("Nothing run yet in this module.")
    if st.button(f"Reset {module.name.lower()}", key=f"reset_{module.key}"):
        st.session_state.pop(f"state_{module.key}", None)
        st.session_state.pop(f"step_{module.key}", None)
        st.rerun()
st.caption("Research use only. Not a medical device. Passing software tests does not make a model scientifically "
           "or clinically validated.")
