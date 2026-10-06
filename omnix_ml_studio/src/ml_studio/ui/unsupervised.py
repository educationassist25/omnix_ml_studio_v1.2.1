"""Clustering, anomaly detection and time series: step definitions and routing."""
from __future__ import annotations

import streamlit as st

from . import anomaly_ui as AU
from . import clustering_ui as CU
from . import shell as SH
from .components import download_all_figures, page_figures, reset_page_figures
from . import timeseries_ui as TU
from . import unsup_common as UC


def _unsup_steps(S, mod, verbs) -> list[SH.Step]:
    u = S.u
    has, ready, chosen = "ud" in u, "Z" in u, u.get("chosen") is not None
    return [
        SH.Step("data", "Data", "Load a table and choose the features. An optional reference label is kept aside "
                "for evaluation.", UC.data_step, True, has),
        SH.Step("setup", "Setup", "Imputation, scaling and optional PCA, fitted on all samples."
                + (" Contamination sets the expected share of anomalies." if mod is AU else ""),
                UC.setup_step, has, ready),
        SH.Step("compare", "Compare", verbs["compare"], mod.compare_step, ready,
                u.get("lb") is not None or u.get("cmp") is not None),
        SH.Step("create", verbs["create_t"], verbs["create"], mod.create_step, ready, chosen),
        SH.Step("analyze", "Analyze", verbs["analyze"], mod.analyze_step, chosen,
                chosen and any(k in (u.get("analysis") or {}) for k in ("pca", "tsne"))),
        SH.Step("validate", "Validate", verbs["validate"], mod.validate_step, chosen, u.get("stab") is not None),
        SH.Step("explain", "Explain", verbs["explain"], mod.explain_step, chosen,
                chosen and (u.get("analysis") or {}).get("prof") is not None if mod is CU else False),
        SH.Step("deploy", verbs["deploy_t"], verbs["deploy"], mod.deploy_step, chosen, u.get("meta") is not None),
    ]


TEXT = {
    "clustering": dict(compare="Fit many clustering algorithms to the same data and compare internal criteria.",
                       create_t="Create & tune", create="Choose k with a scan of criteria, set parameters, and pick the "
                       "final clustering.", analyze="Maps, sizes and silhouettes of the chosen clustering.",
                       validate="Do the clusters survive resampling? Clusterwise bootstrap Jaccard stability.",
                       explain="Which features differ between clusters (descriptive).",
                       deploy_t="Assign & deploy", deploy="Download assignments and a signed bundle; assign new samples."),
    "anomaly": dict(compare="Fit many detectors and see how far they agree.", create_t="Create",
                    create="Set the detector's parameters and contamination, and pick the final model.",
                    analyze="Score distributions, maps and the most anomalous samples.",
                    validate="Are the same samples flagged when the detector is refitted on subsets?",
                    explain="Which features make a flagged sample unusual.",
                    deploy_t="Score & deploy", deploy="Download scores and a signed bundle; score new samples."),
}


def _ts_steps(S) -> list[SH.Step]:
    u = S.u
    has, ready, chosen = "series" in u, "design" in u, u.get("chosen") is not None
    return [
        SH.Step("data", "Data", "Load a series: a date column and the value to forecast.", TU.data_step, True, has),
        SH.Step("setup", "Setup", "Horizon, seasonal period and rolling-origin validation. Confirming seals the last "
                "horizon as a holdout.", TU.setup_step, has, ready),
        SH.Step("compare", "Compare", "Cross-validate many forecasters on the same rolling origins.", TU.compare_step,
                ready, u.get("lb") is not None),
        SH.Step("create", "Create & tune", "Build one forecaster with chosen parameters and pick the final model.",
                TU.create_step, ready, chosen),
        SH.Step("analyze", "Analyze", "Decomposition, autocorrelation and out-of-sample residuals.", TU.analyze_step,
                chosen, False),
        SH.Step("validate", "Validate", "Forecast the sealed holdout once, with prediction intervals.", TU.validate_step,
                chosen, u.get("holdout") is not None),
        SH.Step("deploy", "Forecast & deploy", "Refit on the whole series, forecast ahead and download a signed bundle.",
                TU.deploy_step, chosen, u.get("final") is not None),
    ]


def steps(S, module) -> list[SH.Step]:
    if module.key == "timeseries":
        return _ts_steps(S)
    mod = CU if module.key == "clustering" else AU
    return _unsup_steps(S, mod, TEXT[module.key])


def render(S, module):
    step = SH.step_rail(module, steps(S, module))
    SH.next_banner(module, steps(S, module), step)
    main, side = SH.workspace()
    reset_page_figures()
    st.session_state["_page_zip_done"] = False
    step.render(S, module, main, side)
    if len(page_figures()) > 1 and not st.session_state.get("_page_zip_done"):
        download_all_figures(main, f"{module.key}_{step.key}", label="Download all figures on this step")
    with main:
        SH.next_button(module, steps(S, module), step)
