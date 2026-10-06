"""Typed configuration objects for a study.

Every analysis is driven by three explicit objects so that settings can be recorded,
exported and reproduced:

* :class:`StudyDesign`      – which columns play which role
* :class:`PreprocessConfig` – learned / unlearned preprocessing steps
* :class:`ValidationConfig` – holdout, CV and nested-CV design
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal, Optional

import yaml

TaskType = Literal["binary", "multiclass", "regression"]


@dataclass
class StudyDesign:
    target: str
    task: TaskType
    features: list[str]
    sample_id: Optional[str] = None
    group: Optional[str] = None          # patient / subject ID for repeated measures
    batch: Optional[str] = None          # exploratory / confounding checks only
    time: Optional[str] = None
    covariates: list[str] = field(default_factory=list)   # clinical covariates (subset of features)
    categorical: list[str] = field(default_factory=list)  # overrides of inferred types
    positive_label: Optional[str] = None                  # binary tasks

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PreprocessConfig:
    numeric_imputation: Literal["median", "mean", "knn", "constant", "none"] = "median"
    numeric_fill_value: float = 0.0
    categorical_imputation: Literal["most_frequent", "constant"] = "most_frequent"
    transform: Literal["none", "log2", "log10", "yeo-johnson", "quantile"] = "none"
    # log transforms need an explicit choice when values <= 0 exist; never silent
    log_offset: Optional[float] = None
    transform_columns: Optional[list] = None        # None = all numeric features
    scaling: Literal["standard", "minmax", "robust", "none"] = "standard"
    max_missing_fraction: Optional[float] = 0.5     # drop features above (fit on train)
    variance_threshold: Optional[float] = 0.0       # drop (near-)constant features
    correlation_threshold: Optional[float] = None   # e.g. 0.95
    univariate_k: Optional[int] = None              # SelectKBest inside folds
    fs_method: Optional[str] = None                 # FeatureSelector method inside folds
    fs_k: int = 20
    imbalance: Literal["none", "class_weight", "oversample", "undersample", "smote"] = "none"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ValidationConfig:
    holdout_fraction: float = 0.2
    cv_strategy: Literal["stratified_kfold", "kfold", "repeated_stratified_kfold",
                         "repeated_kfold", "group_kfold", "stratified_group_kfold"] = "stratified_kfold"
    n_folds: int = 5
    n_repeats: int = 1
    inner_folds: int = 3                 # nested CV
    random_state: int = 42
    n_bootstrap: int = 1000
    n_jobs: int = 1

    def to_dict(self) -> dict:
        return asdict(self)


def load_yaml(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh) or {}
