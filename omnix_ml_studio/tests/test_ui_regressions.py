"""UI regressions found in deployment (Streamlit AppTest), on the module/step navigation."""
import os
import sys

import pytest

pytest.importorskip("streamlit.testing.v1")
sys.path.insert(0, os.path.dirname(__file__))
from ui_flow_helpers import click, errors, goto, new_app, pick  # noqa: E402

M = "classification"


def _to_setup(example):
    at = new_app(900)
    at.radio(key=f"{M}_src").set_value("Example dataset").run()
    at.selectbox(key=f"{M}_example").set_value(example).run()
    click(at, f"{M}_load_ex")
    click(at, f"run_{M}_roles")
    click(at, f"run_{M}_prep")
    goto(at, M, "setup")
    return at


def test_log_without_offset_cannot_be_confirmed_and_models_never_crash():
    at = _to_setup("clinical_plus_800_genes.csv")
    at.selectbox(key=f"{M}_tr").set_value("log2").run()
    at.checkbox(key=f"{M}_confirm").check().run()
    btn = at.button(key=f"run_{M}_setup")
    assert btn.disabled, "an invalid log design must not be confirmable"
    assert any("offset" in e for e in errors(at))
    at.checkbox(key=f"{M}_offon").check().run()
    assert not at.button(key=f"run_{M}_setup").disabled
    click(at, f"run_{M}_setup")
    assert at.session_state[f"state_{M}"].design_confirmed
    goto(at, M, "compare")
    click(at, f"{M}_pick_none")
    pick(at, "lr")
    click(at, f"run_{M}_compare")
    assert len(at.session_state[f"state_{M}"].comparison.leaderboard) == 2      # lr + dummy


def test_compare_shows_reason_when_every_model_fails():
    from ml_studio.core.config import PreprocessConfig
    at = _to_setup("clinical_plus_800_genes.csv")
    at.checkbox(key=f"{M}_confirm").check().run()
    click(at, f"run_{M}_setup")
    at.session_state[f"state_{M}"].prep_cfg = PreprocessConfig(transform="log2")   # force past the UI guard
    goto(at, M, "compare")
    click(at, f"{M}_pick_none")
    pick(at, "lr")
    click(at, f"run_{M}_compare")
    assert any("No model could be trained" in e for e in errors(at))


def test_locked_steps_cannot_be_opened_and_modules_keep_separate_state():
    at = new_app(300)
    locked = [b for b in at.button if b.key and b.key.startswith(f"step_{M}_") and b.key.endswith("_locked")]
    assert len(locked) == 7 and all(b.disabled for b in locked)
    at.radio(key=f"{M}_src").set_value("Example dataset").run()
    click(at, f"{M}_load_ex")
    click(at, "mod_clustering_")
    assert at.session_state["state_clustering"].df is None
    assert at.session_state[f"state_{M}"].df is not None
