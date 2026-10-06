"""Helpers to drive the Streamlit app with AppTest (used by test_ui_workflow.py)."""
from __future__ import annotations

import os

from streamlit.testing.v1 import AppTest

os.environ.setdefault("OMNIX_SYNC_JOBS", "1")      # AppTest cannot poll background threads


def new_app(timeout=600):
    import os
    at = AppTest.from_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app.py"),
                           default_timeout=timeout)
    at.run()
    return at


def btn(at, prefix):
    """Button whose key starts with prefix (step buttons carry a status suffix)."""
    for b in at.button:
        if b.key and b.key.startswith(prefix):
            return b
    raise KeyError(f"no button {prefix}; have {[b.key for b in at.button]}")


def click(at, prefix):
    btn(at, prefix).click().run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def goto(at, module, step):
    return click(at, f"step_{module}_{step}_")


def errors(at):
    return [e.value for e in at.error]


def pick(at, model, module="classification", on=True):
    """Tick (or untick) one algorithm in the model picker."""
    cb = [c for c in at.checkbox if c.key and c.key.startswith(f"{module}_pick_cb_{model}_")][0]
    (cb.check() if on else cb.uncheck()).run()
    assert not at.exception, [e.value for e in at.exception]
    return at
