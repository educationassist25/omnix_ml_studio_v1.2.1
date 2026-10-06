"""Dataset loading and data-quality assessment.

Nothing here modifies or deletes data. :func:`assess_quality` returns a
:class:`QualityReport` listing issues and *suggested* exclusions; the caller (UI or
script) decides what to apply, and applied exclusions are recorded.
"""
from __future__ import annotations

import re

import csv
import io
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd

from .config import StudyDesign

Severity = Literal["info", "warning", "critical"]
MAX_UPLOAD_MB = 200


class DataLoadError(ValueError):
    """Raised for unreadable, empty or oversized files."""


# --------------------------------------------------------------------------- loading
def load_table(source: Union[str, os.PathLike, bytes, io.BytesIO], filename: Optional[str] = None,
               max_mb: float = MAX_UPLOAD_MB, sheet: Union[int, str] = 0) -> pd.DataFrame:
    """Read CSV / TSV / XLSX into a DataFrame.

    Duplicate column names are *not* silently renamed: they are reported in
    ``df.attrs["duplicate_columns"]`` (pandas would otherwise mangle them to ``x.1``).
    """
    if isinstance(source, (str, os.PathLike)):
        filename = filename or str(source)
        size = os.path.getsize(source)
        with open(source, "rb") as fh:
            raw = fh.read()
    else:
        raw = source.getvalue() if isinstance(source, io.BytesIO) else bytes(source)
        size = len(raw)
    if size == 0:
        raise DataLoadError("The file is empty.")
    if size > max_mb * 1024 * 1024:
        raise DataLoadError(f"File is {size / 1e6:.0f} MB; the limit is {max_mb} MB.")
    name = (filename or "").lower()
    if name.endswith((".xlsx", ".xlsm")):
        import openpyxl
        df = pd.read_excel(io.BytesIO(raw), sheet_name=sheet, engine="openpyxl")
        wb = openpyxl.load_workbook(io.BytesIO(raw), read_only=True)
        ws = wb.worksheets[sheet] if isinstance(sheet, int) else wb[sheet]
        header = [str(v) for v in next(ws.iter_rows(min_row=1, max_row=1, values_only=True), ())
                  if v is not None]
        wb.close()
    elif name.endswith((".csv", ".tsv", ".txt")) or not name:
        text = raw.decode("utf-8-sig", errors="replace")
        sep = "\t" if name.endswith((".tsv", ".txt")) else ","
        if not name:
            sep = "\t" if text.split("\n", 1)[0].count("\t") > text.split("\n", 1)[0].count(",") else ","
        header = next(csv.reader(io.StringIO(text.split("\n", 1)[0]), delimiter=sep), [])
        df = pd.read_csv(io.StringIO(text), sep=sep)
    else:
        raise DataLoadError(f"Unsupported file type: {filename}. Use CSV, TSV or XLSX.")
    if df.shape[0] == 0 or df.shape[1] == 0:
        raise DataLoadError("The file has no data rows or no columns.")
    dups = [c for c, n in Counter(header).items() if n > 1]
    df.attrs["duplicate_columns"] = dups
    df.columns = [str(c) for c in df.columns]
    return df


# --------------------------------------------------------------------------- inference
def infer_task(y: pd.Series, max_classes: int = 10) -> str:
    y = y.dropna()
    if not pd.api.types.is_numeric_dtype(y) or y.nunique() <= max_classes:
        return "binary" if y.nunique() == 2 else "multiclass"
    return "regression"


def infer_types(df: pd.DataFrame, features: list[str], overrides: Optional[list[str]] = None,
                max_levels: int = 20) -> tuple[list[str], list[str]]:
    """Split features into (numeric, categorical). ``overrides`` forces categorical."""
    overrides = set(overrides or [])
    num, cat = [], []
    for c in features:
        s = df[c]
        if c in overrides or not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            cat.append(c)
        else:
            num.append(c)
    return num, cat


def suggest_roles(df: pd.DataFrame) -> dict:
    """Heuristic guesses for sample-ID / group / batch / time / target columns.
    The user always confirms; these only pre-fill the selectors.

    Target priority: exact outcome-like name > outcome-like word at a name boundary >
    a low-cardinality last column (the common "outcome last" layout). Covariate-like
    names such as ``smoking_status`` or ``vital_status`` are not preferred."""
    out = {"sample_id": None, "group": None, "batch": None, "time": None, "target": None}
    exact = ("target", "outcome", "label", "class", "y", "response", "diagnosis", "event", "subtype",
             "group", "status", "recurrence", "relapse", "phenotype")
    words = ("target", "outcome", "label", "response", "diagnosis", "recurrence", "relapse", "event",
             "subtype", "phenotype", "class")
    best, best_score = None, 0
    for pos, col in enumerate(df.columns):
        name = col.lower()
        unique = df[col].nunique() == len(df)
        if out["sample_id"] is None and unique and (name in ("sample_id", "sample", "id", "sampleid")
                                                    or name.endswith("_id")):
            out["sample_id"] = col
            continue
        toks = [t for t in re.split(r"[^a-z0-9]+", name) if t]
        nun = df[col].nunique()
        if out["group"] is None and not unique and any(t in ("patient", "patients", "subject", "donor", "individual")
                                                       or t.startswith("patient") for t in toks):
            out["group"] = col
            continue
        # whole words only: "run" must not match the gene RUNX3, "plate" must not match platelet_count
        if out["batch"] is None and 1 < nun <= max(2, len(df) // 2) and \
                any(t in ("batch", "plate", "run", "lane", "flowcell", "chip") or re.fullmatch(r"(batch|plate|run)\d*", t)
                    for t in toks):
            out["batch"] = col
            continue
        if out["time"] is None and name in ("time", "visit", "day", "week", "timepoint", "time_point"):
            out["time"] = col
            continue
        parts = name.replace("-", "_").split("_")
        score = 0
        if name in exact:
            score = 3
        elif any(w in parts for w in words):
            score = 2
        if pos == len(df.columns) - 1 and df[col].nunique() <= 10:
            score = max(score, 1) + 0.5          # outcome-last convention
        if score > best_score:
            best, best_score = col, score
    out["target"] = best
    return out


# --------------------------------------------------------------------------- report
@dataclass
class Issue:
    code: str
    severity: Severity
    message: str
    columns: list[str] = field(default_factory=list)
    rows: list = field(default_factory=list)


@dataclass
class QualityReport:
    n_samples: int
    n_features: int
    n_numeric: int
    n_categorical: int
    issues: list[Issue] = field(default_factory=list)
    feature_missing: Optional[pd.Series] = None
    suggested_feature_exclusions: dict[str, str] = field(default_factory=dict)  # col -> reason
    suggested_sample_exclusions: dict = field(default_factory=dict)             # row -> reason

    @property
    def critical(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "critical"]

    def summary_table(self) -> pd.DataFrame:
        return pd.DataFrame([{"severity": i.severity, "check": i.code, "message": i.message,
                              "n_columns": len(i.columns), "n_rows": len(i.rows)}
                             for i in self.issues])

    def add(self, code, severity, message, columns=(), rows=()):
        self.issues.append(Issue(code, severity, message, list(columns), list(rows)))


def _cramers_v(a: pd.Series, b: pd.Series) -> float:
    ct = pd.crosstab(a, b)
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return 0.0
    from scipy.stats import chi2_contingency
    chi2 = chi2_contingency(ct, correction=False)[0]
    n = ct.values.sum()
    return float(np.sqrt(chi2 / (n * (min(ct.shape) - 1))))


def assess_quality(df: pd.DataFrame, design: StudyDesign,
                   max_missing_fraction: float = 0.5,
                   nzv_freq_ratio: float = 95 / 5, nzv_unique_pct: float = 10.0,
                   leakage_threshold: float = 0.98) -> QualityReport:
    """Run all data-quality checks. Never modifies ``df``."""
    feats = [c for c in design.features if c in df.columns]
    num, cat = infer_types(df, feats, design.categorical)
    rep = QualityReport(len(df), len(feats), len(num), len(cat))

    if len(df) == 0:
        rep.add("empty", "critical", "Dataset has no rows.")
        return rep
    missing_cols = [c for c in design.features if c not in df.columns]
    if missing_cols:
        rep.add("unknown_columns", "critical", f"{len(missing_cols)} selected feature(s) not found.",
                missing_cols)
    if design.target not in df.columns:
        rep.add("no_target", "critical", f"Target column '{design.target}' not found.")
        return rep
    if design.target in feats:
        rep.add("target_in_features", "critical", "The target column is also listed as a feature.",
                [design.target])

    # duplicate column names (from loader)
    dups = df.attrs.get("duplicate_columns", [])
    if dups:
        rep.add("duplicate_columns", "critical",
                f"Duplicate column names in the file: {', '.join(dups[:10])}. "
                "Rename them before analysis.", dups)

    # sample IDs
    if design.sample_id and design.sample_id in df.columns:
        sid = df[design.sample_id]
        d = sid[sid.duplicated(keep=False)]
        if len(d):
            rep.add("duplicate_ids", "critical",
                    f"{d.nunique()} sample ID(s) occur more than once ({len(d)} rows). "
                    "If these are repeated measures, set a group/patient column.", rows=list(d.index))
        if sid.isna().any():
            rep.add("missing_ids", "warning", f"{int(sid.isna().sum())} rows have no sample ID.",
                    rows=list(sid.index[sid.isna()]))

    # target
    y = df[design.target]
    ymiss = y.isna()
    if ymiss.any():
        rep.add("missing_target", "warning",
                f"{int(ymiss.sum())} rows have no target value; they cannot be used for "
                "training or evaluation.", rows=list(y.index[ymiss]))
        for r in y.index[ymiss]:
            rep.suggested_sample_exclusions[r] = "missing target"
    yv = y.dropna()
    if design.task in ("binary", "multiclass"):
        counts = yv.value_counts()
        if design.task == "binary" and len(counts) != 2:
            rep.add("target_levels", "critical",
                    f"Binary task but the target has {len(counts)} distinct values.")
        if len(counts) < 2:
            rep.add("single_class", "critical", "The target has only one class.")
        else:
            frac = counts.min() / counts.sum()
            if frac < 0.05:
                rep.add("imbalance", "warning",
                        f"Severe class imbalance: smallest class '{counts.idxmin()}' is "
                        f"{frac:.1%} ({counts.min()} samples). Accuracy will be misleading; "
                        "prefer balanced accuracy, MCC, PR-AUC.")
            elif frac < 0.2:
                rep.add("imbalance", "info",
                        f"Class imbalance: smallest class '{counts.idxmin()}' is {frac:.1%}.")
            if counts.min() < 10:
                rep.add("small_class", "warning",
                        f"Class '{counts.idxmin()}' has only {counts.min()} samples; "
                        "CV estimates will be unstable.")
    else:
        if not pd.api.types.is_numeric_dtype(yv):
            rep.add("target_dtype", "critical", "Regression target is not numeric.")

    # sample size
    n_eff = len(yv)
    if design.group and design.group in df.columns:
        n_eff = df.loc[yv.index, design.group].nunique()
    if n_eff < 30:
        rep.add("sample_size", "warning",
                f"Only {n_eff} independent sampling units. Performance estimates will have "
                "wide uncertainty; prefer repeated CV and report confidence intervals.")
    if len(feats) > 5 * n_eff:
        rep.add("high_dimensional", "info",
                f"High-dimensional data: {len(feats)} features vs {n_eff} independent units. "
                "Feature selection must be done inside CV folds (the engine enforces this).")

    X = df[feats]
    # missingness
    miss = X.isna().mean()
    rep.feature_missing = miss
    if miss.any():
        rep.add("missing_values", "info",
                f"{int((miss > 0).sum())} feature(s) contain missing values "
                f"({X.isna().values.mean():.1%} of cells overall).", list(miss.index[miss > 0]))
    hi = miss[miss > max_missing_fraction]
    if len(hi):
        rep.add("excessive_missing", "warning",
                f"{len(hi)} feature(s) are missing in more than {max_missing_fraction:.0%} of samples.",
                list(hi.index))
        for c in hi.index:
            rep.suggested_feature_exclusions[c] = f"missing {hi[c]:.0%}"
    rmiss = X.isna().mean(axis=1)
    bad_rows = rmiss[rmiss > max_missing_fraction]
    if len(bad_rows):
        rep.add("sample_missing", "warning",
                f"{len(bad_rows)} sample(s) are missing more than {max_missing_fraction:.0%} of features.",
                rows=list(bad_rows.index))

    if num:
        Xn = X[num]
        # dtype sanity: object columns that are mostly numeric
        vals = Xn.to_numpy(dtype=float, na_value=np.nan)
        inf_cols = [num[j] for j in np.where(np.isinf(vals).any(axis=0))[0]]
        if inf_cols:
            rep.add("infinite", "critical", f"{len(inf_cols)} feature(s) contain ±infinity.", inf_cols)
        std = np.nanstd(np.where(np.isinf(vals), np.nan, vals), axis=0)
        const = [num[j] for j in np.where((std == 0) | np.isnan(std))[0]]
        if const:
            rep.add("constant", "warning", f"{len(const)} constant feature(s) carry no information.",
                    const)
            for c in const:
                rep.suggested_feature_exclusions.setdefault(c, "constant")
        # near-zero variance (caret definition)
        nzv = []
        # distinct values per column in one sorted pass; only low-uniqueness columns can be NZV,
        # so the (slow) per-column value_counts runs for those alone
        Vf = np.where(np.isfinite(vals), vals, np.nan)
        Ss = np.sort(Vf, axis=0)
        n_ok = (~np.isnan(Vf)).sum(axis=0)
        changes = ((np.diff(Ss, axis=0) != 0) & ~np.isnan(Ss[1:])).sum(axis=0) if len(Ss) > 1 else np.zeros(len(num))
        distinct = np.where(n_ok > 0, 1 + changes, 0)
        cand = {num[j] for j in np.where(100 * distinct / np.maximum(n_ok, 1) < nzv_unique_pct)[0]}
        for c in num:
            if c in const or c not in cand:
                continue
            vc = Xn[c].value_counts(dropna=True)
            if len(vc) < 2:
                continue
            ratio = vc.iloc[0] / vc.iloc[1]
            uniq = 100 * len(vc) / max(Xn[c].notna().sum(), 1)
            if ratio > nzv_freq_ratio and uniq < nzv_unique_pct:
                nzv.append(c)
        if nzv:
            rep.add("near_zero_variance", "info",
                    f"{len(nzv)} near-zero-variance feature(s) (one value dominates).", nzv)
        neg = [num[j] for j in np.where((np.nan_to_num(vals, nan=1.0) <= 0).any(axis=0))[0]]
        if neg:
            rep.add("nonpositive", "info",
                    f"{len(neg)} feature(s) contain zeros or negative values; log transforms "
                    "need an explicit offset choice.", neg)

    # object columns that look numeric but contain text -> invalid dtype
    for c in cat:
        s = X[c].dropna()
        if s.dtype == object and len(s):
            conv = pd.to_numeric(s, errors="coerce")
            if 0.8 <= conv.notna().mean() < 1.0:
                bad = s[conv.isna()].unique()[:5]
                rep.add("invalid_dtype", "warning",
                        f"'{c}' is mostly numeric but has non-numeric entries "
                        f"(e.g. {', '.join(map(str, bad))}). It is being treated as categorical.",
                        [c])
            nlev = s.nunique()
            if nlev > 50 or nlev > 0.5 * len(s):
                rep.add("high_cardinality", "warning",
                        f"Categorical '{c}' has {nlev} levels; it may be an identifier.", [c])
                rep.suggested_feature_exclusions.setdefault(c, "identifier-like")
            rare = s.value_counts()
            rare = rare[rare < 3]
            if 0 < len(rare) < nlev:
                rep.add("rare_levels", "info",
                        f"'{c}' has {len(rare)} rare level(s) with < 3 samples; unseen levels in "
                        "test data are encoded as all-zeros.", [c])

    # repeated samples (identical feature vectors)
    if len(feats):
        h = pd.util.hash_pandas_object(X, index=False)          # fast row hashes, then an exact check
        dup = pd.Series(False, index=X.index)
        maybe = h.duplicated(keep=False)
        if maybe.any():
            dup.loc[maybe[maybe].index] = X.loc[maybe].duplicated(keep=False).to_numpy()
        if dup.any() and not (design.group and design.group in df.columns):
            rep.add("repeated_samples", "warning",
                    f"{int(dup.sum())} rows have identical feature values. If they are technical "
                    "replicates, set a group column so they stay in the same fold.",
                    rows=list(X.index[dup]))

    # possible target leakage: a single feature that (almost) perfectly predicts y
    leak = _leakage_scan(df.loc[yv.index], feats, num, design, leakage_threshold)
    notes = getattr(_leakage_scan, "notes", {})
    if leak:
        rep.add("target_leakage", "critical",
                "Single features almost perfectly separate the outcome: leakage, or genuinely strong markers? "
                "These features alone predict the "
                f"target: {', '.join(f'{k} ({v:.3f}' + (f'; {notes[k]}' if k in notes else '') + ')' for k, v in list(leak.items())[:8])}. "
                "Check whether they are measured after, or derived from, the outcome. Strong biology (e.g. tumour vs "
                "normal) is also possible; then expect near-perfect performance.",
                list(leak))
        for c in leak:
            rep.suggested_feature_exclusions.setdefault(c, "possible target leakage")

    # batch confounding
    if design.batch and design.batch in df.columns and design.task != "regression":
        v = _cramers_v(df.loc[yv.index, design.batch], yv)
        sev = "critical" if v > 0.8 else "warning" if v > 0.4 else "info"
        rep.add("batch_confounding", sev,
                f"Association between batch and outcome: Cramér's V = {v:.2f}. "
                + ("Batch is strongly confounded with the outcome — models may learn batch, "
                   "not biology." if v > 0.4 else "No strong confounding detected."),
                [design.batch])
    elif design.batch and design.batch in df.columns:
        from scipy.stats import kruskal
        groups = [g.values for _, g in yv.groupby(df.loc[yv.index, design.batch]) if len(g) > 1]
        if len(groups) > 1:
            p = kruskal(*groups).pvalue
            rep.add("batch_confounding", "warning" if p < 0.01 else "info",
                    f"Outcome differs between batches (Kruskal–Wallis p = {p:.2g}).", [design.batch])

    # groups
    if design.group and design.group in df.columns:
        g = df[design.group]
        sizes = g.value_counts()
        rep.add("groups", "info",
                f"{g.nunique()} groups; {int((sizes > 1).sum())} have repeated measurements "
                f"(max {sizes.max()} per group). Grouped CV will keep each group in one fold.")
        if design.task != "regression":
            mixed = df.loc[yv.index].groupby(design.group)[design.target].nunique()
            if (mixed > 1).any():
                rep.add("group_label_mix", "info",
                        f"{int((mixed > 1).sum())} group(s) contain more than one outcome class.")
    return rep


def _leakage_scan(df, feats, num, design, thr) -> dict[str, float]:
    from sklearn.metrics import roc_auc_score
    y = df[design.target]
    out, notes = {}, {}
    _leakage_scan.notes = notes
    if design.task == "binary":
        lv = sorted(y.unique(), key=str)
        yb = (y == lv[-1]).astype(int).values
        V = df[num].to_numpy(dtype=float)
        full = np.isfinite(V).all(axis=0)
        if full.any() and 0 < yb.sum() < len(yb):          # vectorised AUC via ranks (Mann–Whitney)
            from scipy.stats import rankdata
            R = rankdata(V[:, full], axis=0)
            n1, n0 = yb.sum(), len(yb) - yb.sum()
            auc = (R[yb == 1].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)
            for c, a in zip(np.asarray(num)[full], np.maximum(auc, 1 - auc)):
                if a >= thr:
                    out[c] = float(a)
        for j in np.where(~full)[0]:                        # columns with missing values
            x = V[:, j]
            ok = np.isfinite(x)
            if ok.sum() < 10 or len(np.unique(yb[ok])) < 2:
                continue
            a = roc_auc_score(yb[ok], x[ok])
            a = max(a, 1 - a)
            if a >= thr:
                out[num[j]] = a
    elif design.task == "regression":
        yv = y.values.astype(float)
        for c in num:
            x = df[c].values.astype(float)
            ok = np.isfinite(x) & np.isfinite(yv)
            if ok.sum() < 10 or np.std(x[ok]) == 0:
                continue
            r = abs(np.corrcoef(x[ok], yv[ok])[0, 1])
            if r >= thr:
                out[c] = r
    else:   # multiclass
        # (a) a numeric feature that alone separates one class from all others (e.g. pack-years = 0
        #     exactly for never-smokers): definitional for that class, so the model learns it trivially
        from scipy.stats import rankdata
        V = df[num].to_numpy(dtype=float) if num else np.empty((len(df), 0))
        full = np.isfinite(V).all(axis=0) if V.size else np.zeros(0, bool)
        if full.any():
            R = rankdata(V[:, full], axis=0)
            names = np.asarray(num)[full]
            for cls in sorted(y.dropna().unique(), key=str):
                yb = (y == cls).to_numpy()
                n1, n0 = int(yb.sum()), int((~yb).sum())
                if n1 < 5 or n0 < 5:
                    continue
                auc = (R[yb].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)
                for c, a in zip(names, np.maximum(auc, 1 - auc)):
                    if a >= thr and a > out.get(c, 0):
                        out[c] = float(a)
                        notes[c] = f"separates class '{cls}' from the rest"
        # (b) a low-cardinality feature whose values map (almost) 1:1 to classes
        for c in feats:
            s = df[c]
            nu = s.nunique()
            if nu <= 1 or nu >= 0.5 * len(s) or nu > 50:   # continuous / ID-like: not applicable
                continue
            ct = pd.crosstab(s, y)
            purity = ct.max(axis=1).sum() / ct.values.sum()
            if purity >= thr and purity > out.get(c, 0):
                out[c] = float(purity)
                notes[c] = "its levels map almost 1:1 to classes"
    _leakage_scan.notes = notes
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def apply_exclusions(df: pd.DataFrame, design: StudyDesign, drop_features: list[str],
                     drop_rows: list) -> tuple[pd.DataFrame, StudyDesign, dict]:
    """Apply user-confirmed exclusions and return an audit record."""
    from dataclasses import replace
    keep = [f for f in design.features if f not in set(drop_features)]
    out = df.drop(index=[r for r in drop_rows if r in df.index])
    audit = {"dropped_features": list(drop_features), "dropped_rows": [str(r) for r in drop_rows],
             "n_samples_before": len(df), "n_samples_after": len(out),
             "n_features_before": len(design.features), "n_features_after": len(keep)}
    return out, replace(design, features=keep,
                        covariates=[c for c in design.covariates if c in keep],
                        categorical=[c for c in design.categorical if c in keep]), audit


# --------------------------------------------------------------------------- TCGA barcodes
TCGA_SAMPLE_TYPES = {"01": "Primary tumour", "02": "Recurrent tumour", "03": "Blood-derived cancer",
                     "05": "Additional primary", "06": "Metastatic", "07": "Additional metastatic",
                     "10": "Blood normal", "11": "Solid tissue normal", "12": "Buccal normal",
                     "14": "Bone marrow normal"}


def tcga_info(ids: pd.Series) -> Optional[dict]:
    """Patient structure of TCGA barcodes (TCGA-TSS-PATIENT-SAMPLETYPE…), or None if not TCGA.

    The first 12 characters identify the patient; characters 14–15 the sample type (01 tumour,
    11 normal, …). Several samples from one patient must stay together in every split."""
    s = ids.dropna().astype(str)
    if s.empty or s.str.match(r"^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-\d{2}").mean() < 0.9:
        return None
    patient = ids.astype(str).str[:12]
    stype = ids.astype(str).str[13:15]
    per = patient.value_counts()
    return {"patient": patient, "n_patients": int(per.size), "n_samples": int(len(ids)),
            "n_multi": int((per > 1).sum()),
            "types": {f"{k} ({TCGA_SAMPLE_TYPES.get(k, 'other')})": int(v) for k, v in stype.value_counts().items()}}
