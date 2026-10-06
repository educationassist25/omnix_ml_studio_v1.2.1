"""Per-session state. Nothing is cached across users (no st.cache on user data)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from ..core.config import PreprocessConfig, StudyDesign, ValidationConfig

# downstream results invalidated when an upstream stage changes
STAGES = ["data", "design", "split", "models"]
_OWNED = {
    "data": ["df", "filename", "source"],
    "design": ["design", "quality", "audit", "data"],
    "split": ["split", "ledger", "design_confirmed"],
    "models": ["comparison", "tuning", "chosen", "final_model", "holdout", "nested", "external",
               "stability", "fstats", "shap", "perm", "created", "oof", "bundle", "bundle_meta", "predictions",
               "deploy_model", "deploy_ood", "analysis"],
}


@dataclass
class AppState:
    df: Optional[pd.DataFrame] = None
    filename: Optional[str] = None
    source: Optional[str] = None
    design: Optional[StudyDesign] = None
    quality: Any = None
    audit: Optional[dict] = None
    data: Any = None                       # PreparedData after exclusions
    prep_cfg: PreprocessConfig = field(default_factory=PreprocessConfig)
    val_cfg: ValidationConfig = field(default_factory=ValidationConfig)
    design_confirmed: bool = False
    split: Any = None                      # HoldoutSplit (sealed)
    ledger: Any = None
    comparison: Any = None
    tuning: dict = field(default_factory=dict)
    chosen: Optional[dict] = None          # {"key", "params", "source"}
    final_model: Any = None
    holdout: Optional[dict] = None
    nested: Any = None
    external: Any = None
    stability: Any = None                  # feature-selection stability (development rows)
    fstats: Any = None                     # univariate statistics (development rows)
    shap: Any = None
    perm: Any = None
    created: Any = None                    # name -> {"key", "params", "cv": DataFrame, "how"}
    oof: Any = None                        # cache: name -> (y, y_pred, proba)
    analysis: Any = None
    bundle: Any = None
    bundle_meta: Any = None
    deploy_model: Any = None
    deploy_ood: Any = None
    predictions: Any = None
    history: list = field(default_factory=list)
    u: dict = field(default_factory=dict)          # unsupervised / time-series module results

    def invalidate(self, stage: str):
        """Clear results owned by ``stage``'s downstream stages (and the stage itself,
        except for 'data', whose raw upload is kept)."""
        start = STAGES.index(stage)
        for st in STAGES[start:]:
            for name in _OWNED[st]:
                if st == "data":
                    continue
                setattr(self, name, False if name == "design_confirmed" else None)
        self.tuning = {}
        if stage in ("data", "design"):
            self.u = {}

    def log(self, kind: str, summary: str):
        import time
        self.history.append({"time": time.strftime("%H:%M:%S"), "analysis": kind, "summary": summary})


def get_state(st, module: str = "app") -> AppState:
    key = "app" if module == "app" else f"state_{module}"
    if key not in st.session_state:
        st.session_state[key] = AppState()
    return st.session_state[key]
