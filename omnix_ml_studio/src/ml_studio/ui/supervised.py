"""Classification and regression: the eight-step supervised workflow."""
from __future__ import annotations

import streamlit as st

from . import shell as SH
from .components import download_all_figures, page_figures, reset_page_figures
from .sup_data import data_step, setup_step
from .sup_eval import deploy_step, explain_step, validate_step
from .sup_model import analyze_step, compare_step, create_step


def steps(S) -> list[SH.Step]:
    has_data = S.data is not None
    ready = has_data and S.design_confirmed and S.split is not None
    chosen = ready and S.chosen is not None
    return [
        SH.Step("data", "Data", "Load a table, assign column roles and resolve data-quality issues.",
                data_step, True, has_data),
        SH.Step("setup", "Setup", "Preprocessing, feature filtering and the validation design. Confirming seals "
                "the holdout before any model is trained.", setup_step, has_data, ready),
        SH.Step("compare", "Compare", "Cross-validate many algorithms on identical folds of the development set.",
                compare_step, ready, S.comparison is not None),
        SH.Step("create", "Create & tune", "Build one model with default, custom or searched hyperparameters, "
                "then choose the final model.", create_step, ready, S.chosen is not None),
        SH.Step("analyze", "Analyze", "Diagnostic plots for the chosen model from out-of-fold predictions.",
                analyze_step, chosen, chosen and S.oof is not None),
        SH.Step("validate", "Validate", "Unbiased performance: the sealed holdout, nested cross-validation, or an "
                "external cohort.", validate_step, chosen, S.holdout is not None or S.nested is not None),
        SH.Step("explain", "Explain", "How the fitted model uses its inputs: SHAP, permutation importance and "
                "partial dependence.", explain_step, chosen, S.shap is not None or S.perm is not None),
        SH.Step("deploy", "Predict & deploy", "Finalize, download a signed bundle and predict new samples with "
                "out-of-distribution flags.", deploy_step, chosen, S.bundle_meta is not None),
    ]


def render(S, module):
    st_list = steps(S)
    step = SH.step_rail(module, st_list)
    SH.next_banner(module, steps(S), step)
    main, side = SH.workspace()
    reset_page_figures()
    st.session_state["_page_zip_done"] = False
    step.render(S, module, main, side)
    if len(page_figures()) > 1 and not st.session_state.get("_page_zip_done"):
        download_all_figures(main, f"{module.key}_{step.key}", label="Download all figures on this step")
    with main:
        SH.next_button(module, steps(S), step)
