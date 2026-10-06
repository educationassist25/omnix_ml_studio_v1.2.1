"""Regression model registry — the full PyCaret algorithm list (same IDs)."""
from __future__ import annotations

import importlib.util

from scipy.stats import loguniform, randint, uniform
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import (ARDRegression, BayesianRidge, ElasticNet, HuberRegressor, Lars, Lasso,
                                  LassoLars, LinearRegression, OrthogonalMatchingPursuit,
                                  PassiveAggressiveRegressor, RANSACRegressor, Ridge, TheilSenRegressor)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from .classification import ModelSpec

FAMILIES = ["Linear", "Robust linear", "Tree ensembles", "Gradient boosting", "Kernel & neighbours",
            "Neural network", "Baseline"]


def _lgbm(rs):
    from lightgbm import LGBMRegressor
    return LGBMRegressor(random_state=rs, verbose=-1, n_jobs=1)


def _xgb(rs):
    from xgboost import XGBRegressor
    return XGBRegressor(random_state=rs, n_jobs=1, verbosity=0, tree_method="hist")


def _cat(rs):
    from .wide_boost import WideCatBoostRegressor
    return WideCatBoostRegressor(random_state=rs, verbose=0, thread_count=1, iterations=300, allow_writing_files=False)


L, RL, TE, GB, KN, NN, BL = FAMILIES
_ALL = [
    ModelSpec("lr", "Linear Regression", lambda rs: LinearRegression(), {},
              "Unregularised; unstable when p approaches n.", family=L),
    ModelSpec("ridge", "Ridge", lambda rs: Ridge(random_state=rs), {"model__alpha": loguniform(1e-3, 1e3)}, family=L),
    ModelSpec("lasso", "Lasso", lambda rs: Lasso(max_iter=20000, random_state=rs),
              {"model__alpha": loguniform(1e-4, 1e1)}, family=L),
    ModelSpec("en", "Elastic Net", lambda rs: ElasticNet(max_iter=20000, random_state=rs),
              {"model__alpha": loguniform(1e-4, 1e1), "model__l1_ratio": uniform(0.05, 0.9)}, family=L),
    ModelSpec("lar", "Least Angle Regression", lambda rs: Lars(random_state=rs), {}, family=L),
    ModelSpec("llar", "Lasso Least Angle", lambda rs: LassoLars(random_state=rs),
              {"model__alpha": loguniform(1e-4, 1e1)}, family=L),
    ModelSpec("omp", "Orthogonal Matching Pursuit", lambda rs: OrthogonalMatchingPursuit(), {}, family=L),
    ModelSpec("br", "Bayesian Ridge", lambda rs: BayesianRidge(), {}, family=L),
    ModelSpec("ard", "ARD Regression", lambda rs: ARDRegression(), {}, family=L, speed="medium"),
    ModelSpec("par", "Passive Aggressive", lambda rs: PassiveAggressiveRegressor(random_state=rs, max_iter=2000),
              {"model__C": loguniform(1e-3, 1e1)}, family=RL),
    ModelSpec("ransac", "RANSAC", lambda rs: RANSACRegressor(random_state=rs), {}, family=RL),
    ModelSpec("tr", "Theil–Sen", lambda rs: TheilSenRegressor(random_state=rs, n_jobs=1), {},
              "Robust but slow with many features.", family=RL, speed="slow"),
    ModelSpec("huber", "Huber", lambda rs: HuberRegressor(max_iter=2000),
              {"model__epsilon": uniform(1.1, 1.0), "model__alpha": loguniform(1e-5, 1e-1)}, family=RL),
    ModelSpec("rf", "Random Forest", lambda rs: RandomForestRegressor(n_estimators=300, random_state=rs, n_jobs=1),
              {"model__max_depth": [None, 3, 5, 10], "model__min_samples_leaf": randint(1, 10),
               "model__max_features": [1.0, "sqrt", 0.3]}, family=TE, speed="medium"),
    ModelSpec("et", "Extra Trees", lambda rs: ExtraTreesRegressor(n_estimators=300, random_state=rs, n_jobs=1),
              {"model__max_depth": [None, 3, 5, 10], "model__min_samples_leaf": randint(1, 10)},
              family=TE, speed="medium"),
    ModelSpec("dt", "Decision Tree", lambda rs: DecisionTreeRegressor(random_state=rs),
              {"model__max_depth": randint(1, 10), "model__min_samples_leaf": randint(1, 20)}, family=TE),
    ModelSpec("ada", "AdaBoost", lambda rs: AdaBoostRegressor(random_state=rs),
              {"model__n_estimators": randint(25, 300), "model__learning_rate": loguniform(0.01, 2)},
              family=GB, speed="medium"),
    ModelSpec("gbr", "Gradient Boosting", lambda rs: GradientBoostingRegressor(random_state=rs),
              {"model__learning_rate": loguniform(0.01, 0.3), "model__n_estimators": randint(50, 300),
               "model__max_depth": randint(1, 5), "model__subsample": uniform(0.6, 0.4)}, family=GB, speed="medium"),
    ModelSpec("svm", "Support Vector Regression", lambda rs: SVR(),
              {"model__C": loguniform(1e-2, 1e3), "model__epsilon": loguniform(1e-3, 1)}, family=KN),
    ModelSpec("kr", "Kernel Ridge", lambda rs: KernelRidge(kernel="rbf"),
              {"model__alpha": loguniform(1e-3, 1e1), "model__gamma": loguniform(1e-4, 1e0)}, family=KN),
    ModelSpec("knn", "K-Nearest Neighbours", lambda rs: KNeighborsRegressor(),
              {"model__n_neighbors": randint(3, 30), "model__weights": ["uniform", "distance"]}, family=KN),
    ModelSpec("mlp", "MLP (neural network)",
              lambda rs: MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, early_stopping=True, random_state=rs),
              {"model__alpha": loguniform(1e-5, 1e-1), "model__hidden_layer_sizes": [(50,), (100,), (100, 50)]},
              family=NN, speed="medium"),
    ModelSpec("dummy", "Dummy (mean)", lambda rs: DummyRegressor(strategy="mean"), {},
              "Baseline: predicts the training mean.", family=BL),
]
REGISTRY: dict[str, ModelSpec] = {m.key: m for m in _ALL}
for _mod, _spec in [
    ("lightgbm", ModelSpec("lightgbm", "LightGBM", _lgbm,
                           {"model__learning_rate": loguniform(0.01, 0.3), "model__n_estimators": randint(50, 400),
                            "model__num_leaves": randint(4, 32), "model__min_child_samples": randint(5, 30)},
                           family=GB)),
    ("xgboost", ModelSpec("xgboost", "XGBoost", _xgb,
                          {"model__learning_rate": loguniform(0.01, 0.3), "model__n_estimators": randint(50, 400),
                           "model__max_depth": randint(2, 7)}, family=GB)),
    ("catboost", ModelSpec("catboost", "CatBoost", _cat,
                           {"model__learning_rate": loguniform(0.01, 0.3), "model__depth": randint(3, 8)},
                           family=GB, speed="medium")),
]:
    if importlib.util.find_spec(_mod) is not None:
        REGISTRY[_spec.key] = _spec

DEFAULT = [k for k in ["dummy", "ridge", "lasso", "en", "br", "rf", "et", "lightgbm", "xgboost", "svm", "knn"]
           if k in REGISTRY]


def get(key: str) -> ModelSpec:
    if key not in REGISTRY:
        raise KeyError(f"Unknown or unavailable regressor '{key}'. Available: {list(REGISTRY)}")
    return REGISTRY[key]
