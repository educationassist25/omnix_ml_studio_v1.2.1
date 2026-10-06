"""Classification model registry — the full PyCaret algorithm list (same IDs), plus L1 logistic.

Each entry: display name, family, factory(random_state) -> unfitted estimator, a small
hyperparameter search space (keys refer to the pipeline step ``model__``), whether it
outputs probabilities, and a relative speed hint. Optional libraries (XGBoost, CatBoost,
LightGBM) appear only when importable.
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

FAMILIES = ["Linear", "Tree ensembles", "Gradient boosting", "Kernel & neighbours",
            "Probabilistic", "Neural network", "Baseline"]


@dataclass
class ModelSpec:
    key: str
    name: str
    factory: Callable[[int], object]
    space: dict = field(default_factory=dict)
    notes: str = ""
    family: str = "Other"
    proba: bool = True
    speed: str = "fast"            # fast | medium | slow (relative, for the picker)
    pycaret: bool = True           # part of PyCaret's default list


def _lgbm(rs):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(random_state=rs, verbose=-1, n_jobs=1)


def _xgb(rs):
    from xgboost import XGBClassifier
    return XGBClassifier(random_state=rs, n_jobs=1, verbosity=0, tree_method="hist")


def _cat(rs):
    from .wide_boost import WideCatBoostClassifier
    return WideCatBoostClassifier(random_state=rs, verbose=0, thread_count=1, iterations=300,
                              allow_writing_files=False)



class ScaledGPC(GaussianProcessClassifier):
    """Gaussian process classifier whose default RBF length scale starts at √p.

    With the library default (length scale 1) and standardised data, distances between samples
    grow like √p, so with thousands of features every kernel value is ~0 and the model predicts
    the prior (AUC 0.5). Starting at √p puts the kernel in a sensible range; the optimiser still
    tunes it. A user-given kernel is used unchanged."""

    def fit(self, X, y):
        if self.kernel is not None:
            return super().fit(X, y)
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel
        p = X.shape[1]
        self.kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=float(np.sqrt(p)),
                                                              length_scale_bounds=(1e-2 * np.sqrt(p), 1e2 * np.sqrt(p)))
        try:
            return super().fit(X, y)
        finally:
            self.kernel = None


class AdaptiveLDA(LinearDiscriminantAnalysis):
    """LDA with Ledoit–Wolf shrinkage when features are few; the SVD solver when they are many.

    The shrinkage solvers build a p × p covariance matrix (10,000 genes ≈ 0.8 GB and many minutes per
    fit). Above ``WIDE`` features the SVD solver is used instead, which never forms that matrix.
    """
    WIDE = 1500

    def fit(self, X, y):
        p = X.shape[1]
        if p <= self.WIDE or self.solver == "svd":
            return super().fit(X, y)
        est = LinearDiscriminantAnalysis(solver="svd", priors=self.priors, tol=self.tol).fit(X, y)
        for k, v in vars(est).items():
            if k.endswith("_"):
                setattr(self, k, v)
        self.wide_solver_ = "svd"
        return self

_ALL = [
    ModelSpec("lr", "Logistic Regression", lambda rs: LogisticRegression(max_iter=5000, random_state=rs),
              {"model__C": loguniform(1e-3, 1e2)}, family="Linear"),
    ModelSpec("lr_l1", "Logistic Regression (L1 / lasso)",
              lambda rs: LogisticRegression(penalty="l1", solver="liblinear", max_iter=2000, random_state=rs),
              {"model__C": loguniform(1e-2, 1e2)}, "Sparse; useful when features greatly outnumber samples.",
              family="Linear", pycaret=False),
    ModelSpec("ridge", "Ridge Classifier", lambda rs: RidgeClassifier(random_state=rs),
              {"model__alpha": loguniform(1e-3, 1e3)}, "No probabilities: AUC uses the decision function.",
              family="Linear", proba=False),
    ModelSpec("svm", "SVM (linear kernel, SGD)",
              lambda rs: SGDClassifier(loss="hinge", max_iter=5000, tol=1e-4, random_state=rs),
              {"model__alpha": loguniform(1e-6, 1e-1)}, "No probabilities: AUC uses the decision function.",
              family="Linear", proba=False),
    ModelSpec("lda", "Linear Discriminant Analysis",
              lambda rs: AdaptiveLDA(solver="lsqr", shrinkage="auto"), {},
              "Ledoit–Wolf shrinkage; switches to the SVD solver above 1,500 features (no p × p matrix).",
              family="Linear"),
    ModelSpec("rf", "Random Forest", lambda rs: RandomForestClassifier(n_estimators=300, random_state=rs, n_jobs=1),
              {"model__max_depth": [None, 3, 5, 10], "model__min_samples_leaf": randint(1, 10),
               "model__max_features": ["sqrt", "log2", 0.3]}, family="Tree ensembles", speed="medium"),
    ModelSpec("et", "Extra Trees", lambda rs: ExtraTreesClassifier(n_estimators=300, random_state=rs, n_jobs=1),
              {"model__max_depth": [None, 3, 5, 10], "model__min_samples_leaf": randint(1, 10),
               "model__max_features": ["sqrt", "log2", 0.3]}, family="Tree ensembles", speed="medium"),
    ModelSpec("dt", "Decision Tree", lambda rs: DecisionTreeClassifier(random_state=rs),
              {"model__max_depth": randint(1, 10), "model__min_samples_leaf": randint(1, 20)},
              family="Tree ensembles"),
    ModelSpec("ada", "AdaBoost", lambda rs: AdaBoostClassifier(random_state=rs, algorithm="SAMME"),
              {"model__n_estimators": randint(25, 300), "model__learning_rate": loguniform(0.01, 2)},
              family="Gradient boosting", speed="medium"),
    ModelSpec("gbc", "Gradient Boosting", lambda rs: GradientBoostingClassifier(random_state=rs),
              {"model__learning_rate": loguniform(0.01, 0.3), "model__n_estimators": randint(50, 300),
               "model__max_depth": randint(1, 5), "model__subsample": uniform(0.6, 0.4)},
              family="Gradient boosting", speed="medium"),
    ModelSpec("rbfsvm", "SVM (RBF kernel)", lambda rs: SVC(probability=True, random_state=rs),
              {"model__C": loguniform(1e-2, 1e2), "model__gamma": ["scale", "auto"]},
              "Probabilities via internal Platt scaling.", family="Kernel & neighbours", speed="medium"),
    ModelSpec("knn", "K-Nearest Neighbours", lambda rs: KNeighborsClassifier(),
              {"model__n_neighbors": randint(3, 30), "model__weights": ["uniform", "distance"]},
              family="Kernel & neighbours"),
    ModelSpec("gpc", "Gaussian Process", lambda rs: ScaledGPC(random_state=rs, n_jobs=1), {},
              "Cost grows with n³; avoid above ~2,000 samples.", family="Kernel & neighbours", speed="slow"),
    ModelSpec("nb", "Naive Bayes", lambda rs: GaussianNB(), {"model__var_smoothing": loguniform(1e-10, 1e-3)},
              family="Probabilistic"),
    ModelSpec("qda", "Quadratic Discriminant Analysis", lambda rs: QuadraticDiscriminantAnalysis(reg_param=0.1),
              {"model__reg_param": uniform(0.0, 1.0)}, "Unstable when p approaches n.", family="Probabilistic"),
    ModelSpec("mlp", "MLP (neural network)",
              lambda rs: MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, early_stopping=True,
                                       random_state=rs),
              {"model__alpha": loguniform(1e-5, 1e-1), "model__hidden_layer_sizes": [(50,), (100,), (100, 50)],
               "model__learning_rate_init": loguniform(1e-4, 1e-2)}, family="Neural network", speed="medium"),
    ModelSpec("dummy", "Dummy (prior)", lambda rs: DummyClassifier(strategy="prior"), {},
              "Baseline that ignores features. Every model should beat it.", family="Baseline"),
]
_OPTIONAL = [
    ("lightgbm", ModelSpec("lightgbm", "LightGBM", _lgbm,
                           {"model__learning_rate": loguniform(0.01, 0.3), "model__n_estimators": randint(50, 400),
                            "model__num_leaves": randint(4, 32), "model__min_child_samples": randint(5, 30)},
                           family="Gradient boosting")),
    ("xgboost", ModelSpec("xgboost", "XGBoost", _xgb,
                          {"model__learning_rate": loguniform(0.01, 0.3), "model__n_estimators": randint(50, 400),
                           "model__max_depth": randint(2, 7), "model__subsample": uniform(0.6, 0.4)},
                          family="Gradient boosting")),
    ("catboost", ModelSpec("catboost", "CatBoost", _cat,
                           {"model__learning_rate": loguniform(0.01, 0.3), "model__depth": randint(3, 8)},
                           family="Gradient boosting", speed="medium")),
]

REGISTRY: dict[str, ModelSpec] = {m.key: m for m in _ALL}
for _mod, _spec in _OPTIONAL:
    if importlib.util.find_spec(_mod) is not None:
        REGISTRY[_spec.key] = _spec

DEFAULT = [k for k in ["dummy", "lr", "rf", "et", "lightgbm", "xgboost", "rbfsvm", "knn", "nb"] if k in REGISTRY]


def get(key: str) -> ModelSpec:
    if key not in REGISTRY:
        raise KeyError(f"Unknown or unavailable classifier '{key}'. Available: {list(REGISTRY)}")
    return REGISTRY[key]
