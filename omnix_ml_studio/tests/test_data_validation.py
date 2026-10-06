import io

import numpy as np
import pandas as pd
import pytest
from conftest import design_for

from ml_studio.core import synthetic
from ml_studio.core.config import StudyDesign
from ml_studio.core.data_validation import (DataLoadError, apply_exclusions, assess_quality, suggest_roles,
                                            infer_task, load_table)


def codes(rep):
    return {i.code: i for i in rep.issues}


# ---------------------------------------------------------------- loading
def test_load_csv_tsv_xlsx_roundtrip(tmp_path, binary_df):
    p = tmp_path / "d.csv"; binary_df.to_csv(p, index=False)
    assert load_table(p).shape == binary_df.shape
    p = tmp_path / "d.tsv"; binary_df.to_csv(p, sep="\t", index=False)
    assert load_table(p).shape == binary_df.shape
    p = tmp_path / "d.xlsx"; binary_df.head(50).to_excel(p, index=False)
    assert load_table(p).shape == (50, binary_df.shape[1])


def test_empty_and_oversize_files_rejected(tmp_path):
    p = tmp_path / "e.csv"; p.write_bytes(b"")
    with pytest.raises(DataLoadError):
        load_table(p)
    with pytest.raises(DataLoadError):
        load_table(io.BytesIO(b"a,b\n" + b"1,2\n" * 1000), "x.csv", max_mb=0.001)
    with pytest.raises(DataLoadError):
        load_table(io.BytesIO(b"a,b\n"), "x.csv")
    with pytest.raises(DataLoadError):
        load_table(io.BytesIO(b"abc"), "x.pdf")


def test_duplicate_column_names_detected():
    df = load_table(io.BytesIO(b"id,a,a,y\n1,1,2,0\n2,3,4,1\n"), "d.csv")
    assert df.attrs["duplicate_columns"] == ["a"]
    rep = assess_quality(df, design_for(df, "y", "binary", exclude=("id",)))
    assert codes(rep)["duplicate_columns"].severity == "critical"


def test_infer_task():
    assert infer_task(pd.Series([0, 1, 1, 0])) == "binary"
    assert infer_task(pd.Series(list("abcabc"))) == "multiclass"
    assert infer_task(pd.Series(np.linspace(0, 1, 50))) == "regression"


# ---------------------------------------------------------------- checks
def test_clean_dataset_has_no_critical(binary_df):
    rep = assess_quality(binary_df, design_for(binary_df, "outcome", "binary", sample_id="sample_id"))
    assert rep.critical == []
    assert rep.n_samples == 300 and rep.n_categorical == 2


def test_problem_dataset_flags_everything():
    df = synthetic.binary(n=120, seed=9)
    df.loc[[0, 1], "sample_id"] = "DUP"
    df.loc[[5, 6], "outcome"] = np.nan
    df["const"] = 1.0
    df["mostly_missing"] = np.where(np.arange(120) < 100, np.nan, 1.0)
    df.loc[3, "f0002"] = np.inf
    df["nzv"] = np.r_[np.zeros(117), [1, 2, 3]]
    df["num_text"] = [str(i) for i in range(115)] + ["n.d."] * 5
    df = synthetic.with_leak(df.dropna(subset=["outcome"]).reset_index(drop=True)
                             .pipe(lambda d: d), "outcome")
    df.loc[[0, 1], "sample_id"] = "DUP"
    d = design_for(df, "outcome", "binary", sample_id="sample_id")
    rep = assess_quality(df, d)
    c = codes(rep)
    for k in ("duplicate_ids", "constant", "excessive_missing", "infinite", "near_zero_variance",
              "invalid_dtype", "target_leakage"):
        assert k in c, k
    assert c["target_leakage"].columns[0] == "leaky_marker"
    assert "leaky_marker" in rep.suggested_feature_exclusions
    assert "const" in rep.suggested_feature_exclusions


def test_missing_target_suggested_not_dropped(binary_df):
    df = binary_df.copy(); df.loc[[1, 2, 3], "outcome"] = np.nan
    rep = assess_quality(df, design_for(df, "outcome", "binary"))
    assert set(rep.suggested_sample_exclusions) == {1, 2, 3}
    assert len(df) == 300                      # data untouched


def test_imbalance_and_small_sample():
    df = synthetic.imbalanced(n=200, minority=0.03)
    rep = assess_quality(df, design_for(df, "event", "binary"))
    assert codes(rep)["imbalance"].severity == "warning"
    small = synthetic.binary(n=20, seed=3)
    assert "sample_size" in codes(assess_quality(small, design_for(small, "outcome", "binary")))


def test_batch_confounding_detected():
    df = synthetic.batch_confounded()
    rep = assess_quality(df, design_for(df, "outcome", "binary", exclude=("sample_id", "batch"),
                                        batch="batch"))
    assert codes(rep)["batch_confounding"].severity in ("warning", "critical")


def test_repeated_samples_and_groups(grouped_df):
    d = design_for(grouped_df, "response", "binary",
                   exclude=("sample_id", "patient_id", "visit"), group="patient_id")
    c = codes(assess_quality(grouped_df, d))
    assert "groups" in c and "60 groups" in c["groups"].message
    dup = pd.concat([grouped_df.head(5), grouped_df.head(5)], ignore_index=True)
    dup = pd.concat([dup, grouped_df.iloc[5:60]], ignore_index=True)
    d2 = design_for(dup, "response", "binary", exclude=("sample_id", "patient_id", "visit"))
    assert "repeated_samples" in codes(assess_quality(dup, d2))


def test_multiclass_and_regression_leakage(multiclass_df, regression_df):
    leak = synthetic.with_leak(regression_df, "response")
    assert "target_leakage" not in codes(assess_quality(
        regression_df, design_for(regression_df, "response", "regression")))
    leak["leaky_marker"] = leak["response"] * 2 + 1
    assert "target_leakage" in codes(assess_quality(leak, design_for(leak, "response", "regression")))
    mc = multiclass_df.copy(); mc["copy_of_y"] = mc["subtype"]
    assert "target_leakage" in codes(assess_quality(mc, design_for(mc, "subtype", "multiclass")))


def test_apply_exclusions_is_audited(binary_df):
    d = design_for(binary_df, "outcome", "binary")
    out, d2, audit = apply_exclusions(binary_df, d, ["f0001"], [0, 1])
    assert "f0001" not in d2.features and len(out) == 298
    assert audit["n_samples_before"] == 300 and audit["dropped_features"] == ["f0001"]
    assert len(binary_df) == 300


def test_hdlss_quality_is_fast(omics_df):
    import time
    t = time.time()
    rep = assess_quality(omics_df, design_for(omics_df, "group", "binary"))
    assert time.time() - t < 20
    assert "high_dimensional" in codes(rep) and "nonpositive" in codes(rep)


def test_target_guess_prefers_outcome_over_covariate_status():
    """Regression: 'smoking_status' was guessed as the target instead of 'recurrence_2yr'."""
    from ml_studio.core.data_validation import suggest_roles
    df = pd.DataFrame({"patient_id": range(20), "smoking_status": list("NCF") * 6 + ["N", "C"],
                       "age": range(20), "recurrence_2yr": [0, 1] * 10})
    assert suggest_roles(df)["target"] == "recurrence_2yr"
    df2 = df.rename(columns={"recurrence_2yr": "x9"})
    assert suggest_roles(df2)["target"] == "x9"          # low-cardinality last column


def test_multiclass_quality_fast_on_wide_continuous_data():
    import time
    df = synthetic.omics_hdlss(n=300, p=1500)
    df["subtype"] = np.random.default_rng(0).choice(list("ABC"), 300)
    t = time.time()
    assess_quality(df, design_for(df, "subtype", "multiclass", exclude=("sample_id", "group")))
    assert time.time() - t < 10


def test_multiclass_one_class_separator_is_flagged():
    """A numeric feature that is exactly 0 for one class only (pack-years for never-smokers)."""
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(0)
    n = 300
    status = rng.choice(["Never", "Former", "Current"], n)
    df = pd.DataFrame({"status": status, "pack_years": np.where(status == "Never", 0.0, rng.uniform(2, 80, n)),
                       "age": rng.normal(60, 10, n), "noise": rng.normal(0, 1, n)})
    rep = assess_quality(df, StudyDesign(target="status", task="multiclass", features=["pack_years", "age", "noise"]))
    leak = [i for i in rep.issues if i.code == "target_leakage"]
    assert leak and leak[0].columns == ["pack_years"] and "Never" in leak[0].message
    assert "age" not in rep.suggested_feature_exclusions


def test_role_guesses_use_whole_words():
    """Gene symbols and clinical names must not be mistaken for batch or patient columns."""
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame({"sample_id": [f"S{i}" for i in range(n)], "RUNX3": rng.normal(size=n),
                       "platelet_count": rng.normal(250, 30, n), "PLATR1": rng.normal(size=n),
                       "outcome": rng.choice(["a", "b"], n)})
    g = suggest_roles(df)
    assert g["batch"] is None and g["group"] is None and g["target"] == "outcome"
    df["seq_run"] = rng.choice(["r1", "r2", "r3"], n)
    df["patient_id"] = np.repeat(np.arange(n // 2), 2)
    g = suggest_roles(df)
    assert g["batch"] == "seq_run" and g["group"] == "patient_id"
