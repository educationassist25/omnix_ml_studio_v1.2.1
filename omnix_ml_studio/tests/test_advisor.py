"""Automatic analysis suggestion on the example datasets and edge cases."""
import os

import numpy as np
import pandas as pd
import pytest

from ml_studio.core.advisor import profile_dataset

EX = os.path.join(os.path.dirname(__file__), "..", "examples")


@pytest.mark.parametrize("file,module,outcome", [
    ("clinical_plus_800_genes.csv", "classification", "recurrence_2yr"),
    ("binary_classification.csv", "classification", "outcome"),
    ("multiclass_classification.csv", "classification", "subtype"),
    ("regression.csv", "regression", "response"),
    ("clustering.csv", "clustering", "true_subtype"),
    ("anomaly.csv", "anomaly", "is_anomaly"),
    ("monthly_series.csv", "timeseries", "value"),
    ("grouped_omics.csv", "classification", "response"),
    ("omics_hdlss.csv", "classification", "group"),
])
def test_examples_get_the_intended_suggestion(file, module, outcome):
    pr = profile_dataset(pd.read_csv(os.path.join(EX, file)))
    top = pr.suggestions[0]
    assert (top.module, top.outcome) == (module, outcome), [(s.module, s.outcome) for s in pr.suggestions]
    assert top.reasons and pr.summary


def test_roles_and_cautions():
    pr = profile_dataset(pd.read_csv(os.path.join(EX, "grouped_omics.csv")))
    assert pr.roles["sample_id"] == "sample_id" and pr.roles["group"] == "patient_id"
    assert any("repeated" in c for c in pr.suggestions[0].cautions)
    pr = profile_dataset(pd.read_csv(os.path.join(EX, "omics_hdlss.csv")))
    assert any("More features" in c for c in pr.suggestions[0].cautions)
    pr = profile_dataset(pd.read_csv(os.path.join(EX, "clinical_plus_800_genes.csv")))
    assert len(pr.omics_features) == 800 and "pack_years" in pr.clinical_features


def test_no_label_suggests_unsupervised():
    df = pd.read_csv(os.path.join(EX, "clustering.csv")).drop(columns=["true_subtype"])
    mods = [s.module for s in profile_dataset(df).suggestions]
    assert mods[0] == "clustering" and "anomaly" in mods


def test_definitional_feature_is_flagged():
    df = pd.read_csv(os.path.join(EX, "clinical_plus_800_genes.csv")).drop(columns=["recurrence_2yr"])
    df = df[[c for c in df.columns if c != "smoking_status"] + ["smoking_status"]]
    top = profile_dataset(df).suggestions[0]
    assert top.outcome == "smoking_status" and any("pack_years" in c for c in top.cautions)


def test_imbalance_caution_and_ids_never_outcomes():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"record_id": [f"R{i}" for i in range(400)], "x1": rng.normal(size=400),
                       "x2": rng.normal(size=400), "outcome": (rng.random(400) < 0.08).astype(int)})
    pr = profile_dataset(df)
    top = pr.suggestions[0]
    assert top.outcome == "outcome" and any("smallest class" in c for c in top.cautions)
    assert all(c.column != "record_id" for c in pr.candidates)
