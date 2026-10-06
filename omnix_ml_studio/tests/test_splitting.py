import numpy as np
import pytest

from ml_studio.core.config import ValidationConfig
from ml_studio.core.splitting import holdout_split, make_cv, recommend_strategy

GROUPED = ["group_kfold", "stratified_group_kfold"]


def _y_groups(n_groups=50, per=4, seed=0):
    rng = np.random.default_rng(seed)
    yg = rng.binomial(1, 0.4, n_groups)
    return np.repeat(yg, per), np.repeat(np.arange(n_groups), per)


@pytest.mark.parametrize("strategy", GROUPED)
@pytest.mark.parametrize("repeats", [1, 3])
def test_groups_never_cross_folds(strategy, repeats):
    y, g = _y_groups()
    cv = make_cv(ValidationConfig(cv_strategy=strategy, n_folds=5, n_repeats=repeats), "binary", True)
    n = 0
    for tr, te in cv.split(np.zeros(len(y)), y, g):
        assert not set(g[tr]) & set(g[te])
        n += 1
    assert n == 5 * repeats


def test_grouped_holdout_keeps_groups_together():
    y, g = _y_groups()
    for task in ("binary", "regression"):
        s = holdout_split(y.astype(float) if task == "regression" else y, task, 0.2, g, 1)
        assert not set(g[s.train_idx]) & set(g[s.test_idx])
        assert 0.1 < len(s.test_idx) / len(y) < 0.35
        assert len(set(s.train_idx) | set(s.test_idx)) == len(y)


def test_stratified_holdout_preserves_prevalence():
    y = np.r_[np.zeros(400), np.ones(100)].astype(int)
    s = holdout_split(y, "binary", 0.2, None, 0)
    assert len(s.test_idx) == 100 and y[s.test_idx].mean() == pytest.approx(0.2)


def test_ungrouped_strategy_with_groups_is_refused():
    with pytest.raises(ValueError, match="leak"):
        make_cv(ValidationConfig(cv_strategy="stratified_kfold"), "binary", has_groups=True)


def test_recommendations():
    assert recommend_strategy("binary", 500, True)[0] == "stratified_group_kfold"
    assert recommend_strategy("regression", 500, True)[0] == "group_kfold"
    s, k, r, _ = recommend_strategy("binary", 100, False, min_class=6)
    assert s == "repeated_stratified_kfold" and k == 5 and r == 5
    assert recommend_strategy("binary", 100, False, min_class=4)[1] == 4


def test_cv_reproducible():
    y = np.random.default_rng(0).binomial(1, .5, 100)
    a = [te.tolist() for _, te in make_cv(ValidationConfig(random_state=3), "binary", False).split(y, y)]
    b = [te.tolist() for _, te in make_cv(ValidationConfig(random_state=3), "binary", False).split(y, y)]
    assert a == b
