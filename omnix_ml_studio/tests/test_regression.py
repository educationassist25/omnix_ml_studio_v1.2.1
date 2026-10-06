from conftest import design_for

from ml_studio.core import regression
from ml_studio.core.config import PreprocessConfig, ValidationConfig
from ml_studio.core.model_comparison import compare_models
from ml_studio.core.preprocessing import prepare_xy


def test_every_registered_regressor_runs(regression_df):
    d = prepare_xy(regression_df, design_for(regression_df, "response", "regression"))
    r = compare_models(d, "regression", list(regression.REGISTRY), PreprocessConfig(),
                       ValidationConfig(cv_strategy="kfold"))
    assert r.failures == {}
    lb = r.leaderboard.set_index("key")
    assert lb.loc["dummy", "r2_mean"] < 0.05
    assert lb.loc["ridge", "r2_mean"] > 0.6
    assert (lb["rmse_mean"] > 0).all() and (lb["mae_mean"] > 0).all()
    assert "mape_mean" in lb          # y is strictly positive here


def test_mape_omitted_when_zero_target(regression_df):
    df = regression_df.copy(); df.loc[0, "response"] = 0.0
    d = prepare_xy(df, design_for(df, "response", "regression"))
    r = compare_models(d, "regression", ["ridge"], PreprocessConfig(), ValidationConfig(cv_strategy="kfold"))
    assert "mape_mean" not in r.leaderboard
