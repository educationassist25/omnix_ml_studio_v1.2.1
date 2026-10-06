"""Split features into omics (gene / protein / metabolite identifiers) and clinical variables.

If the user marked clinical covariates, those define the clinical group exactly. Otherwise a
naming heuristic is used: gene-style identifiers are upper-case symbols or accessions (TP53,
HLA-DRB1, C20orf27, RP11-156E8.1, ENSG00000141510); clinical variables are usually written in
lower case or with underscores (age, pack_years, tumor_size_cm). The split is shown to the user
and can be corrected by marking clinical covariates in the Data step.
"""
from __future__ import annotations

import re

_GENE = re.compile(r"[A-Z0-9][A-Z0-9\-\.]*(orf\d+[A-Z0-9\-\.]*)?")
_ACCESSION = re.compile(r"(ENS[A-Z]*[GTP]\d{6,}|[NX][MR]_\d+|[OPQ]\d[A-Z0-9]{3}\d|HMDB\d+|C\d{5})(\.\d+)?")


def looks_like_omics(name: str) -> bool:
    n = str(name)
    if _ACCESSION.fullmatch(n):
        return True
    return bool(_GENE.fullmatch(n)) and any(ch.isalpha() for ch in n) and "_" not in n and len(n) <= 30


def split_feature_groups(features: list, covariates: list | None = None) -> dict:
    feats = list(features)
    if covariates:
        cov = set(covariates)
        clinical = [f for f in feats if f in cov]
        source = "clinical covariates marked in the Data step"
    else:
        clinical = [f for f in feats if not looks_like_omics(f)]
        source = "naming pattern (mark clinical covariates in the Data step to set it exactly)"
    cl = set(clinical)
    return {"omics": [f for f in feats if f not in cl], "clinical": clinical, "source": source}
