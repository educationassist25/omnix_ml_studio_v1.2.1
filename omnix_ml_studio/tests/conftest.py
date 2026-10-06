import os
import sys
import warnings

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore", category=UserWarning)

from ml_studio.core import synthetic  # noqa: E402
from ml_studio.core.config import StudyDesign  # noqa: E402


def design_for(df, target, task, exclude=("sample_id",), **kw):
    feats = [c for c in df.columns if c not in (target, *exclude)]
    return StudyDesign(target=target, task=task, features=feats, **kw)


@pytest.fixture(scope="session")
def binary_df():
    return synthetic.binary()


@pytest.fixture(scope="session")
def multiclass_df():
    return synthetic.multiclass()


@pytest.fixture(scope="session")
def regression_df():
    return synthetic.regression()


@pytest.fixture(scope="session")
def omics_df():
    return synthetic.omics_hdlss()


@pytest.fixture(scope="session")
def grouped_df():
    return synthetic.grouped_repeated()
