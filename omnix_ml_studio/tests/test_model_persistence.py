import io
import zipfile

import joblib
import numpy as np
import pytest
from conftest import design_for

from ml_studio.core import model_manager as MM
from ml_studio.core.config import PreprocessConfig, ValidationConfig
from ml_studio.core.explainability import model_input
from ml_studio.core.preprocessing import prepare_xy
from ml_studio.core.validation import fit_final


@pytest.fixture(scope="module")
def fitted(binary_df):
    d = prepare_xy(binary_df, design_for(binary_df, "outcome", "binary"))
    pipe = fit_final(d, np.arange(len(d.y)), "binary", "lightgbm", PreprocessConfig(), ValidationConfig())
    return d, pipe


def test_roundtrip_identical_predictions(fitted):
    d, pipe = fitted
    ood = MM.fit_ood_detector(model_input(pipe, d.X))
    meta = {"task": "binary", "classes": d.classes, "profile": MM.training_profile(d.X, d.numeric, d.categorical),
            "fingerprint": MM.dataset_fingerprint(d.X, d.y)}
    b = MM.make_bundle(pipe, meta, ood)
    m2, ood2, meta2 = MM.load_bundle(b)
    assert np.array_equal(pipe.predict_proba(d.X), m2.predict_proba(d.X))
    assert meta2["classes"] == d.classes and "scikit-learn" in meta2["software"]
    assert MM.read_metadata(b)["fingerprint"] == meta["fingerprint"]


def test_tampered_or_foreign_bundles_are_refused(fitted):
    d, pipe = fitted
    b = MM.make_bundle(pipe, {"task": "binary"})
    zin = zipfile.ZipFile(io.BytesIO(b))
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as z:
        for n in zin.namelist():
            data = zin.read(n)
            if n == "metadata.json":
                data = data.replace(b'"binary"', b'"regression"')
            z.writestr(n, data)
    with pytest.raises(MM.BundleError, match="Signature"):
        MM.load_bundle(out.getvalue())
    raw = io.BytesIO(); joblib.dump({"model": pipe}, raw)               # arbitrary pickle upload
    with pytest.raises(MM.BundleError):
        MM.load_bundle(raw.getvalue())
    out2 = io.BytesIO()
    with zipfile.ZipFile(out2, "w") as z:
        for n in zin.namelist():
            z.writestr(n, zin.read(n))
        z.writestr("extra.py", b"print('x')")
    with pytest.raises(MM.BundleError, match="Unexpected"):
        MM.load_bundle(out2.getvalue())


def test_new_data_checks_and_ood(fitted):
    d, pipe = fitted
    prof = MM.training_profile(d.X, d.numeric, d.categorical)
    new = d.X.head(20).copy()
    new.loc[new.index[:5], "f0001"] = 99.0
    new["stage"] = new["stage"].replace({"III": "IV"})
    c = MM.check_new_data(new.drop(columns=["f0002"]), prof)
    assert c["missing"] == ["f0002"] and "stage" in c["unseen_levels"]
    assert (c["frac_out_of_range"].iloc[:5] > 0).all() and (c["frac_out_of_range"].iloc[5:] == 0).all()
    ood = MM.fit_ood_detector(model_input(pipe, d.X))
    far = d.X.head(3).copy(); far[d.numeric] = 50.0
    _, pct_far = MM.ood_scores(ood, model_input(pipe, far))
    _, pct_in = MM.ood_scores(ood, model_input(pipe, d.X.head(50)))
    assert (pct_far > 99).all() and np.median(pct_in) < 90


def test_fingerprint_sensitive():
    import pandas as pd
    a = pd.DataFrame({"x": [1.0, 2.0]})
    assert MM.dataset_fingerprint(a) != MM.dataset_fingerprint(a.assign(x=[1.0, 2.1]))
