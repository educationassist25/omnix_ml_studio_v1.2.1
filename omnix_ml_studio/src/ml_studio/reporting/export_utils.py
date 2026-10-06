"""Export helpers (CSV / XLSX bytes)."""
from __future__ import annotations

import io

import pandas as pd


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def to_xlsx_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for name, d in sheets.items():
            d = d.copy()
            for c in d.columns:                       # dict/list cells -> str for Excel
                if d[c].map(lambda v: isinstance(v, (dict, list, set))).any():
                    d[c] = d[c].astype(str)
            d.to_excel(xw, sheet_name=str(name)[:31], index=False)
    return buf.getvalue()
