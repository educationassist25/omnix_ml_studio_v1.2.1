"""Background jobs: long computations that survive clicks, reruns and navigation.

Streamlit reruns the whole script whenever the user touches a widget, which interrupts anything
running on the script thread. A job runs on its own thread (the heavy work itself still runs in the
isolated worker processes of :mod:`ml_studio.core.runner`), reports progress and partial results,
and can be stopped. Jobs live in ``st.session_state``, so they are private to one browser session.

``OMNIX_SYNC_JOBS=1`` runs jobs synchronously (used by the automated UI tests).
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import pandas as pd
import streamlit as st


@dataclass
class Job:
    key: str
    label: str
    signature: object = None
    status: str = "running"            # running / done / error / stopped
    done: int = 0
    total: int = 0
    message: str = ""
    partial: list = field(default_factory=list)
    result: object = None
    error: str = ""
    started: float = field(default_factory=time.time)
    finished: float = 0.0
    consumed: bool = False
    _stop: threading.Event = field(default_factory=threading.Event)

    # called from the worker thread ------------------------------------------------
    def progress(self, done: int, total: int, message: str = ""):
        self.done, self.total, self.message = done, total, str(message)

    def add(self, row: dict):
        self.partial.append(row)

    def stopped(self) -> bool:
        return self._stop.is_set()

    # called from the page --------------------------------------------------------
    def stop(self):
        self._stop.set()

    @property
    def running(self) -> bool:
        return self.status == "running"

    @property
    def elapsed(self) -> float:
        return (self.finished or time.time()) - self.started


def _sync() -> bool:
    return os.environ.get("OMNIX_SYNC_JOBS") == "1"


def start(key: str, label: str, fn: Callable, signature=None, **kwargs) -> Job:
    """Run ``fn(job, **kwargs)`` in the background; its return value becomes ``job.result``."""
    old = get(key)
    if old is not None and old.running:
        old.stop()
    job = Job(key, label, signature)

    def run():
        try:
            job.result = fn(job, **kwargs)
            job.status = "stopped" if job.stopped() else "done"
        except Exception as e:  # noqa: BLE001 - shown to the user, never raised into the page
            job.status, job.error = "error", f"{type(e).__name__}: {e}"
        job.finished = time.time()

    st.session_state[f"job_{key}"] = job
    if _sync():
        run()
    else:
        threading.Thread(target=run, name=f"omnix-{key}", daemon=True).start()
    return job


def get(key: str) -> Optional[Job]:
    return st.session_state.get(f"job_{key}")


def running(prefix: str = "") -> list[Job]:
    return [v for k, v in st.session_state.items() if k.startswith(f"job_{prefix}") and isinstance(v, Job) and v.running]


def collect(key: str, apply: Callable[[Job], None], signature=None) -> Optional[Job]:
    """If job ``key`` finished and was not yet applied, apply it (once). Results computed for an
    older setup (different ``signature``) are discarded."""
    job = get(key)
    if job is None or job.running or job.consumed:
        return job
    job.consumed = True
    if signature is not None and job.signature is not None and job.signature != signature:
        job.status, job.error = "error", "the setup changed while this was running; results were discarded"
        return job
    if job.status in ("done", "stopped") and job.result is not None:
        apply(job)
    return job


def _fmt(sec: float) -> str:
    sec = int(sec)
    return f"{sec // 60} min {sec % 60:02d} s" if sec >= 60 else f"{sec} s"


def monitor(key: str, render_partial: Optional[Callable[[Job], None]] = None, every: float = 1.5):
    """Live progress panel for job ``key`` (a fragment that refreshes itself while the job runs)."""
    job = get(key)
    if job is None or not job.running:
        return

    @st.fragment(run_every=every)
    def _panel():
        j = get(key)
        if j is None:
            return
        if not j.running:
            st.rerun()                     # show the final results on the full page
        box = st.container(key=f"jobbox_{key}")
        frac = j.done / j.total if j.total else 0.0
        a, b = box.columns([4, 1], vertical_alignment="center")
        a.progress(min(max(frac, 0.0), 1.0),
                   text=f"{j.label}: {j.done} of {j.total or '…'} finished · {j.message} · {_fmt(j.elapsed)}")
        if b.button("Stop", key=f"jobstop_{key}", use_container_width=True,
                    help="Stops the run; everything finished so far is kept."):
            j.stop()
            j.message = "stopping…"
        box.caption("Runs in the background: you can keep working in other steps or modules. Results appear "
                    "here as each model finishes.")
        if render_partial and j.partial:
            render_partial(j)

    _panel()


def partial_table(job: Job, metric: str, label: str, lower_is_better: bool = False, extra: dict | None = None):
    """Small sorted table of finished rows for the live panel."""
    rows = pd.DataFrame(job.partial)
    if rows.empty or f"{metric}_mean" not in rows:
        return
    rows = rows.sort_values(f"{metric}_mean", ascending=lower_is_better, na_position="last")
    out = pd.DataFrame({"Model": rows["Model"], label: rows[f"{metric}_mean"].round(4),
                        "Time (s)": rows["wall_time_s"].round(1)})
    for k, v in (extra or {}).items():
        if f"{v}_mean" in rows:
            out[k] = rows[f"{v}_mean"].round(4).values
    st.dataframe(out, hide_index=True, width="stretch", height=min(38 + 35 * len(out), 360))


class StopRequested(Exception):
    pass


def stoppable_progress(job: Job, verb: str = "Fitting"):
    """A ``progress(i, n, name)`` callback for in-process loops: updates the job and ends the loop on Stop."""
    def cb(i, n, name):
        if job.stopped():
            raise StopRequested()
        job.progress(i, n, f"{verb} {name}")
    return cb


def run_stoppable(job: Job, fn: Callable, *a, **kw):
    try:
        return fn(*a, **kw)
    except StopRequested:
        return None
