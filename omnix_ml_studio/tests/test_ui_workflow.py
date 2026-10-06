"""End-to-end AppTest runs of every module's full workflow (slow: run with -m ui or the full suite)."""
import os
import sys

import pytest

pytest.importorskip("streamlit.testing.v1")
sys.path.insert(0, os.path.dirname(__file__))
from ui_flow_helpers import click, errors, goto, new_app  # noqa: E402

pytestmark = pytest.mark.ui


def _load(at, mod, example):
    if mod != "classification":
        click(at, f"mod_{mod}_")
    at.radio(key=f"{mod}_src").set_value("Example dataset").run()
    at.selectbox(key=f"{mod}_example").set_value(example).run()
    click(at, f"{mod}_load_ex")
    click(at, f"run_{mod}_roles")


def _plots(at, mod):
    for p in at.radio(key=f"{mod}_plotsel").options:
        at.radio(key=f"{mod}_plotsel").set_value(p).run()
        assert not at.exception, (p, [e.value for e in at.exception])
        assert not errors(at), (p, errors(at))


@pytest.mark.parametrize("mod,example", [("classification", "binary_classification.csv"),
                                         ("classification", "multiclass_classification.csv"),
                                         ("regression", "regression.csv")])
def test_supervised_workflow(mod, example):
    at = new_app()
    _load(at, mod, example)
    click(at, f"run_{mod}_prep")
    goto(at, mod, "setup")
    at.checkbox(key=f"{mod}_confirm").check().run()
    click(at, f"run_{mod}_setup")
    goto(at, mod, "compare")
    click(at, f"run_{mod}_compare")
    S = at.session_state[f"state_{mod}"]
    assert len(S.comparison.leaderboard) >= 5 and not errors(at)
    click(at, f"{mod}_tobest")
    click(at, f"run_{mod}_create")
    at.button(key=f"{mod}_setfinal").click().run()
    assert S.chosen is not None
    goto(at, mod, "analyze")
    _plots(at, mod)
    goto(at, mod, "validate")
    click(at, f"run_{mod}_holdout")
    assert S.holdout is not None and S.ledger.n_evaluations == 1
    goto(at, mod, "explain")
    click(at, f"run_{mod}_shap")
    assert S.shap is not None and not errors(at)
    goto(at, mod, "deploy")
    click(at, f"run_{mod}_finalize")
    assert S.bundle is not None and S.bundle_meta["module"] == mod


@pytest.mark.parametrize("mod,example", [("clustering", "clustering.csv"), ("anomaly", "anomaly.csv"),
                                         ("timeseries", "monthly_series.csv")])
def test_unsupervised_and_forecasting_workflow(mod, example):
    at = new_app()
    _load(at, mod, example)
    goto(at, mod, "setup")
    click(at, f"run_{mod}_setup")
    goto(at, mod, "compare")
    click(at, f"run_{mod}_compare")
    assert not errors(at)
    click(at, f"{mod}_tobest")
    if mod == "clustering":
        click(at, f"{mod}_runscan")
    click(at, f"run_{mod}_create")
    at.button(key=f"{mod}_setfinal").click().run()
    goto(at, mod, "analyze")
    _plots(at, mod)
    goto(at, mod, "validate")
    click(at, f"run_{mod}_stab" if mod != "timeseries" else f"run_{mod}_holdout")
    assert not errors(at)
    if mod != "timeseries":
        goto(at, mod, "explain")
        assert not errors(at)
    goto(at, mod, "deploy")
    click(at, f"run_{mod}_finalize")
    U = at.session_state[f"state_{mod}"].u
    assert U["meta"]["module"] == mod and U["bundle"]
