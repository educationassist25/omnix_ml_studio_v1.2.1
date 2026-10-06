"""Read an uploaded table and suggest which analysis fits it (no Streamlit).

The advisor only *suggests*: it explains its reasoning, lists alternatives, and the user
confirms the module and outcome. Heuristics are transparent and deliberately conservative.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .feature_groups import looks_like_omics

OUTCOME_WORDS = ("outcome", "target", "label", "class", "response", "status", "diagnosis", "recurrence", "relapse",
                 "survival", "event", "death", "progression", "group", "subtype", "grade", "stage", "y")
ANOMALY_WORDS = ("anomaly", "anomalous", "outlier", "fraud", "fault", "defect", "is_abnormal", "abnormal")
CLUSTER_WORDS = ("subtype", "cluster", "true_subtype", "phenotype", "cell_type", "celltype")
ID_WORDS = ("id", "sample", "patient", "subject", "barcode", "accession", "record", "mrn")
TIME_WORDS = ("date", "time", "month", "year", "week", "day", "period", "timestamp")
VALUE_WORDS = ("value", "sales", "count", "demand", "volume", "price", "load", "level", "cases", "y")


@dataclass
class Candidate:
    column: str
    task: str                  # binary / multiclass / regression
    score: float
    reasons: list = field(default_factory=list)
    levels: int = 0
    minority_share: Optional[float] = None


@dataclass
class Suggestion:
    module: str                # classification / regression / clustering / anomaly / timeseries
    confidence: str            # High / Medium / Low
    outcome: Optional[str]
    task: Optional[str]
    reasons: list
    cautions: list = field(default_factory=list)


@dataclass
class DatasetProfile:
    n_rows: int
    n_cols: int
    numeric: list
    categorical: list
    id_columns: list
    group_columns: list
    date_columns: list
    constant: list
    missing_fraction: float
    omics_features: list
    clinical_features: list
    candidates: list
    suggestions: list
    summary: str
    roles: dict


def _is_date(s: pd.Series, name: str) -> bool:
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    if pd.api.types.is_numeric_dtype(s):
        return False
    sample = s.dropna().astype(str).head(200)
    if sample.empty or not sample.str.contains(r"\d").all():
        return False
    parsed = pd.to_datetime(sample, errors="coerce", format="mixed") if hasattr(pd, "to_datetime") else None
    ok = parsed.notna().mean() if parsed is not None else 0
    return ok > 0.9 and (any(w in name.lower() for w in TIME_WORDS) or ok == 1.0)


def _name_has(name: str, words) -> bool:
    n = name.lower()
    toks = set(re.split(r"[^a-z0-9]+", n))
    return any(w in toks or (len(w) > 3 and w in n) for w in words)


def profile_dataset(df: pd.DataFrame) -> DatasetProfile:
    n, p = df.shape
    cols = list(df.columns)
    numeric = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    date_cols = [c for c in cols if _is_date(df[c], str(c))]
    categorical = [c for c in cols if c not in numeric and c not in date_cols]
    nun = df.nunique(dropna=True)
    constant = [c for c in cols if nun[c] <= 1]
    id_cols = [c for c in cols if c not in date_cols and nun[c] >= 0.98 * n and
               (not pd.api.types.is_float_dtype(df[c])) and (_name_has(str(c), ID_WORDS) or c not in numeric)]
    group_cols = [c for c in cols if c not in id_cols and c not in date_cols and _name_has(str(c), ID_WORDS)
                  and 1 < nun[c] < 0.9 * n and nun[c] > 0.05 * n]
    feat_like = [c for c in cols if c not in id_cols + group_cols + date_cols]
    omics = [c for c in feat_like if looks_like_omics(c)]
    clinical = [c for c in feat_like if c not in set(omics)]

    # ---- outcome candidates
    cands = []
    for i, c in enumerate(cols):
        if c in id_cols or c in group_cols or c in date_cols or c in constant:
            continue
        s = df[c].dropna()
        k = int(nun[c])
        reasons, score = [], 0.0
        named = _name_has(str(c), OUTCOME_WORDS)
        if named:
            score += 3; reasons.append("its name suggests an outcome")
        if i == p - 1:
            score += 1.5; reasons.append("it is the last column")
        if looks_like_omics(c) and len(omics) > 20:
            score -= 3                                   # a gene among many is rarely the outcome
        if k == 2:
            task = "binary"
            share = float(s.value_counts(normalize=True).min())
            score += 1.5; reasons.append(f"two values ({', '.join(map(str, sorted(s.unique(), key=str)))})")
        elif 2 < k <= 10 and (c not in numeric or np.allclose(s, np.round(s))):
            task, share = "multiclass", float(s.value_counts(normalize=True).min())
            score += 0.5; reasons.append(f"{k} categories")
        elif c in numeric and k > 15:
            task, share = "regression", None
            reasons.append("continuous numeric values")
        else:
            continue
        if c in categorical and k <= 10 and not named:
            score -= 0.5                                 # categorical descriptors (sex, stage) are often predictors
        cands.append(Candidate(c, task, score, reasons, k, share))
    cands.sort(key=lambda x: -x.score)
    best = cands[0] if cands and cands[0].score >= 2.5 else None

    # ---- suggestions
    sugg: list[Suggestion] = []
    cautions_common = []
    features_n = len(feat_like) - (1 if best else 0)
    if features_n > n:
        cautions_common.append(f"More features ({features_n:,}) than samples ({n:,}): use an in-fold univariate "
                               "pre-filter and nested cross-validation; expect wide uncertainty.")
    if group_cols:
        cautions_common.append(f"'{group_cols[0]}' repeats across rows (repeated measures): it will be used as a group "
                               "so the same subject never appears in both training and test data.")
    if id_cols and not group_cols:
        from .data_validation import tcga_info
        ti = tcga_info(df[id_cols[0]])
        if ti and ti["n_multi"]:
            cautions_common.append(f"'{id_cols[0]}' holds TCGA barcodes: {ti['n_multi']} of {ti['n_patients']} patients "
                                   "have more than one sample (e.g. tumour and matched normal). Group by patient "
                                   "(first 12 characters) so a patient never appears in both training and test data.")
    miss = float(df.isna().values.mean())
    if miss > 0.05:
        cautions_common.append(f"{miss:.0%} of cells are missing; imputation is fitted inside each training fold.")

    regular_series = False
    if date_cols:
        d = pd.to_datetime(df[date_cols[0]], errors="coerce", format="mixed")
        if d.notna().mean() > 0.95 and d.is_unique and n >= 24:
            regular_series = True
    if regular_series and len(numeric) <= 5:
        val = next((c for c in numeric if _name_has(str(c), VALUE_WORDS)), numeric[0] if numeric else None)
        sugg.append(Suggestion("timeseries", "High", val, "forecast",
                               [f"'{date_cols[0]}' is a date column with one row per time point",
                                f"{n} time points and only {len(numeric)} numeric column" + ("s" if len(numeric) != 1 else ""),
                                f"'{val}' is the value to forecast"]))

    anom = next((c for c in cands if c.task == "binary" and _name_has(c.column, ANOMALY_WORDS)), None)
    if anom is not None:
        sugg.append(Suggestion("anomaly", "High", anom.column, None,
                               [f"'{anom.column}' looks like an anomaly flag "
                                f"({(anom.minority_share or 0):.0%} of rows flagged)",
                                "anomaly detection learns from the other columns without using this flag; the flag "
                                "is kept aside to check the result"],
                               ["If the flag is reliable and you want to predict it, Classification is an alternative "
                                "(treat the imbalance)."]))

    clus = next((c for c in cands if c.task in ("multiclass", "binary") and _name_has(c.column, CLUSTER_WORDS)), None)

    if best is not None and best is not anom:
        if best.task in ("binary", "multiclass"):
            cau = list(cautions_common)
            if best.minority_share is not None and best.minority_share < 0.2:
                cau.append(f"The smallest class is {best.minority_share:.0%} of samples: prefer balanced accuracy, "
                           "MCC or PR-AUC, and consider class weights.")
            sugg.append(Suggestion("classification", "High" if best.score >= 4 else "Medium", best.column, best.task,
                                   [f"'{best.column}' is the likely outcome: " + "; ".join(best.reasons),
                                    f"{best.task} classification with {features_n:,} candidate features"], cau))
        else:
            sugg.append(Suggestion("regression", "High" if best.score >= 4 else "Medium", best.column, "regression",
                                   [f"'{best.column}' is the likely outcome: " + "; ".join(best.reasons),
                                    f"regression with {features_n:,} candidate features"], list(cautions_common)))
    reference_like = clus is not None and re.match(r"(true|reference|ref|ground_?truth|known)[_ ]", clus.column.lower())
    if reference_like and sugg and sugg[-1].module == "classification" and sugg[-1].outcome == clus.column:
        sugg[-1].confidence = "Medium"
        sugg[-1].cautions.append(f"The name '{clus.column}' suggests a reference label kept for checking a clustering.")
    if clus is not None:
        sugg.append(Suggestion("clustering", "High" if reference_like else ("Medium" if best is clus else "Low"),
                               clus.column, None,
                               [f"'{clus.column}' looks like known groups: clustering can test whether these groups "
                                "emerge from the features alone (the label is used only to evaluate)"]))
    if not sugg or (best is None and anom is None and not regular_series):
        sugg.append(Suggestion("clustering", "Medium", None, None,
                               ["no column looks like an outcome", f"{len(numeric)} numeric features to group samples by"]))
        sugg.append(Suggestion("anomaly", "Low", None, None, ["without labels, anomaly detection can flag unusual samples"]))
    # secondary alternatives for supervised data
    if best is not None and best.task != "regression" and not any(s.module == "clustering" for s in sugg):
        sugg.append(Suggestion("clustering", "Low", None, None,
                               ["explore whether samples form groups without using the outcome"]))
    # a feature that alone (almost) determines the suggested outcome
    sup = next((x for x in sugg if x.module in ("classification", "regression")), None)
    if sup is not None:
        try:
            from .config import StudyDesign
            from .data_validation import _leakage_scan, infer_types
            feats = [c for c in feat_like if c != sup.outcome]
            num, _ = infer_types(df, feats)
            d0 = df.loc[df[sup.outcome].notna()]
            leak = _leakage_scan(d0, feats, num, StudyDesign(target=sup.outcome, task=sup.task, features=feats), 0.98)
            notes = getattr(_leakage_scan, "notes", {})
            for c, v in list(leak.items())[:3]:
                sup.cautions.append(f"'{c}' alone almost determines '{sup.outcome}' ({v:.3f}"
                                    + (f"; {notes[c]}" if c in notes else "") + "). Check it is not derived from, "
                                    "or measured after, the outcome.")
        except Exception:  # noqa: BLE001 - advice only
            pass
    other = [c for c in cands if best is not None and c is not best and c.score >= 2.5]
    if other:
        sugg[0].cautions.append("Other possible outcome columns: " + ", ".join(f"{c.column} ({c.task})" for c in other[:4])
                                + ". Leave them out of the features if they are measured after the outcome.")
    order = {"High": 0, "Medium": 1, "Low": 2}
    sugg.sort(key=lambda s: order[s.confidence])

    roles = {"sample_id": id_cols[0] if id_cols else None, "group": group_cols[0] if group_cols else None,
             "date": date_cols[0] if date_cols else None, "outcome": sugg[0].outcome}
    top = sugg[0]
    special = set(id_cols + group_cols + date_cols + ([top.outcome] if top.outcome else []))
    n_num = len([c for c in numeric if c not in special])
    n_cat = len([c for c in categorical if c not in special])
    n_clin = len([c for c in clinical if c not in special])
    if top.module in ("classification", "regression"):
        what = f"{MODULE_NAMES[top.module]} of '{top.outcome}'"
    elif top.module == "timeseries":
        what = f"time-series forecasting of '{top.outcome}' over '{date_cols[0]}'"
    else:
        what = MODULE_NAMES[top.module] + (f", with '{top.outcome}' kept aside as the reference label"
                                           if top.outcome else " (no outcome column found)")
    if top.module == "timeseries":
        head = f"{n:,} time points and {p:,} columns. "
    else:
        head = (f"{n:,} samples (rows) and {p:,} columns. Features: {n_num:,} numeric and {n_cat:,} categorical"
                + (f"; {len(omics):,} look like gene/omics identifiers and {n_clin} like clinical variables"
                   if len(omics) >= 10 else "") + ". ")
    miss_txt = ("No missing values. " if miss == 0 else
                ("Under 0.1% of cells are missing. " if miss < 0.001 else f"{miss:.1%} of cells are missing. "))
    summary = (head
               + (f"'{id_cols[0]}' is a unique sample ID. " if id_cols else "")
               + (f"'{group_cols[0]}' groups repeated rows. " if group_cols else "")
               + (f"'{date_cols[0]}' is a date column. " if date_cols else "")
               + miss_txt
               + f"Suggested analysis: {what}.")
    return DatasetProfile(n, p, numeric, categorical, id_cols, group_cols, date_cols, constant, miss, omics, clinical,
                          cands, sugg, summary, roles)


MODULE_NAMES = {"classification": "classification", "regression": "regression", "clustering": "clustering",
                "anomaly": "anomaly detection", "timeseries": "time-series forecasting"}
