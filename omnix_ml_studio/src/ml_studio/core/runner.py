"""Run heavy model fits in isolated worker processes, in parallel, with time and memory limits.

Why: one expensive model (or one that exhausts memory) must never freeze or kill the app. Each
task runs in a worker process. The parent watches every worker: a task that exceeds its time
limit, or a worker whose memory exceeds its share, is stopped and reported, the worker is
replaced, and the remaining tasks carry on. The data are written once to a temporary file and
loaded once per worker, so workers do not receive a copy per task.

Workers are fresh Python subprocesses (no inherited threads or OpenMP state, no re-import of the
app script) that talk to the parent over private pipes. On non-POSIX systems tasks run in-process.
"""
from __future__ import annotations

import os
import pickle
import tempfile
import time
import traceback
from dataclasses import dataclass
from multiprocessing.connection import wait
from typing import Callable, Optional

_PAYLOAD = None


# --------------------------------------------------------------------------- resources
def available_cpus() -> int:
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:
        n = os.cpu_count() or 1
    try:   # cgroup v2 CPU quota (containers, Streamlit Cloud, Render)
        with open("/sys/fs/cgroup/cpu.max") as f:
            q, p = f.read().split()
            if q != "max":
                n = min(n, max(1, int(int(q) / int(p))))
    except (OSError, ValueError):
        pass
    return max(1, n)


def available_memory_mb() -> int:
    import psutil
    avail = psutil.virtual_memory().available / 2 ** 20
    try:   # cgroup v2 memory limit
        with open("/sys/fs/cgroup/memory.max") as f:
            lim = f.read().strip()
        with open("/sys/fs/cgroup/memory.current") as f:
            cur = int(f.read().strip())
        if lim != "max":
            avail = min(avail, (int(lim) - cur) / 2 ** 20)
    except (OSError, ValueError):
        pass
    return int(max(256, avail))


def plan_workers(requested: Optional[int] = None, per_task_mb: int = 700) -> tuple[int, int]:
    """(workers, memory limit per worker in MB) that fit the CPUs and the free memory."""
    cpus = available_cpus()
    usable = int(available_memory_mb() * 0.8)
    n = requested or cpus
    n = max(1, min(n, cpus, usable // max(per_task_mb, 1)))
    return n, max(per_task_mb, usable // n)


# --------------------------------------------------------------------------- worker side
def _serve(rx, tx, payload_path):
    import warnings
    warnings.filterwarnings("ignore")
    global _PAYLOAD
    with open(payload_path, "rb") as f:
        _PAYLOAD = pickle.load(f)
    tx.send(("ready", None, None))
    while True:
        try:
            msg = rx.recv()
        except (EOFError, OSError):
            return
        if msg is None:
            return
        tid, fn, kwargs = msg
        try:
            out = fn(_PAYLOAD, **kwargs)
            tx.send(("ok", tid, out))
        except MemoryError:
            tx.send(("memory", tid, "out of memory"))
        except Exception as e:  # noqa: BLE001
            tx.send(("error", tid, f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}"))


def _subprocess_entry():
    """Entry point of a worker subprocess: argv = [rx_fd, tx_fd, payload_path]."""
    import sys
    from multiprocessing.connection import Connection
    rx = Connection(int(sys.argv[1]), writable=False)
    tx = Connection(int(sys.argv[2]), readable=False)
    _serve(rx, tx, sys.argv[3])


# --------------------------------------------------------------------------- parent side
@dataclass
class TaskResult:
    status: str            # ok / timeout / memory / error / crashed / skipped
    value: object = None
    message: str = ""
    seconds: float = 0.0


class _Worker:
    def __init__(self, payload_path):
        import subprocess
        import sys
        from multiprocessing.connection import Connection
        r1, w1 = os.pipe()          # parent -> child
        r2, w2 = os.pipe()          # child -> parent
        src = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        env = dict(os.environ, PYTHONPATH=src + os.pathsep + os.environ.get("PYTHONPATH", ""),
                   OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONWARNINGS="ignore")
        self.proc = subprocess.Popen(
            [sys.executable, "-c", "from ml_studio.core.runner import _subprocess_entry; _subprocess_entry()",
             str(r1), str(w2), payload_path], pass_fds=(r1, w2), env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.close(r1)
        os.close(w2)
        self.tx = Connection(w1, readable=False)
        self.conn = Connection(r2, writable=False)
        self.ready = False
        self.task = None
        self.started = 0.0

    def alive(self) -> bool:
        return self.proc.poll() is None

    def send(self, msg):
        self.tx.send(msg)

    def rss_mb(self) -> float:
        import psutil
        try:
            p = psutil.Process(self.proc.pid)
            return sum(c.memory_info().rss for c in [p, *p.children(recursive=True)]) / 2 ** 20
        except psutil.Error:
            return 0.0

    def kill(self):
        for c in (self.tx, self.conn):
            try:
                c.close()
            except Exception:  # noqa: BLE001
                pass
        try:
            if self.proc.poll() is None:
                self.proc.kill()
            self.proc.wait(5)
        except Exception:  # noqa: BLE001
            pass


def _run_inprocess(payload, fn, tasks, total_budget_s, progress, should_stop=None, on_done=None):
    out, t0 = {}, time.time()
    for i, (tid, kw) in enumerate(tasks):
        if should_stop and should_stop():
            out[tid] = TaskResult("skipped", message="skipped: stopped by user")
            continue
        if progress:
            progress(i, len(tasks), tid)
        if total_budget_s and time.time() - t0 > total_budget_s:
            out[tid] = TaskResult("skipped", message="skipped: time budget exhausted")
            continue
        t = time.time()
        try:
            out[tid] = TaskResult("ok", fn(payload, **kw), "", time.time() - t)
        except MemoryError:
            out[tid] = TaskResult("memory", None, "out of memory", time.time() - t)
        except Exception as e:  # noqa: BLE001
            out[tid] = TaskResult("error", None, f"{type(e).__name__}: {e}", time.time() - t)
        if on_done:
            on_done(tid, out[tid])
    return out


CRASH_MSG = "the worker stopped unexpectedly (most often: not enough memory)"


def run_isolated(payload: dict, fn: Callable, tasks: list[tuple[str, dict]], workers: Optional[int] = None,
                 timeout_s: Optional[float] = None, mem_limit_mb: Optional[int] = None,
                 total_budget_s: Optional[float] = None,
                 progress: Optional[Callable[[int, int, str], None]] = None,
                 should_stop: Optional[Callable[[], bool]] = None,
                 on_done: Optional[Callable[[str, "TaskResult"], None]] = None) -> dict[str, TaskResult]:
    """Run ``fn(payload, **kwargs)`` for each (task_id, kwargs) in isolated workers.

    ``fn`` must be a module-level function (importable by the workers). Returns task_id → TaskResult.
    ``progress(done, total, task_id)`` is called as tasks start and finish; ``on_done(task_id, result)``
    as each finishes. When ``should_stop()`` becomes true, running workers are stopped and queued
    tasks are reported as skipped.
    """
    if not tasks:
        return {}
    n_workers, auto_mem = plan_workers(workers)
    mem_limit = mem_limit_mb or auto_mem
    n_workers = max(1, min(n_workers, len(tasks)))
    if os.name != "posix":
        return _run_inprocess(payload, fn, tasks, total_budget_s, progress, should_stop, on_done)
    fd, path = tempfile.mkstemp(prefix="omnix_payload_", suffix=".pkl")
    results: dict[str, TaskResult] = {}
    pool: list[_Worker] = []
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        queue = list(tasks)
        t0 = time.time()
        done = 0
        restarts = 0
        pool = [_Worker(path) for _ in range(n_workers)]

        def finish(tid, res):
            nonlocal done
            results[tid] = res
            done += 1
            if on_done:
                try:
                    on_done(tid, res)
                except Exception:  # noqa: BLE001 - a display callback must never break the run
                    pass
            if progress:
                progress(done, len(tasks), tid)

        def replace(i):
            nonlocal restarts, queue
            pool[i].kill()
            restarts += 1
            if restarts > 3 * n_workers + len(tasks):          # workers cannot start at all
                for tid, _ in queue:
                    finish(tid, TaskResult("error", message="worker processes could not start"))
                queue = []
            else:
                pool[i] = _Worker(path)

        while queue or any(w.task for w in pool):
            if should_stop and should_stop():                   # user pressed Stop
                for tid, _ in queue:
                    finish(tid, TaskResult("skipped", message="skipped: stopped by user"))
                queue = []
                for w in pool:
                    if w.task is not None:
                        finish(w.task, TaskResult("skipped", message="stopped by user", seconds=time.time() - w.started))
                        w.task = None
                break
            for w in pool:                                      # hand out work
                if w.task is None and w.ready and queue:
                    if total_budget_s and time.time() - t0 > total_budget_s:
                        for tid, _ in queue:
                            finish(tid, TaskResult("skipped", message="skipped: time budget exhausted"))
                        queue = []
                        break
                    tid, kw = queue.pop(0)
                    w.task, w.started = tid, time.time()
                    try:
                        w.send((tid, fn, kw))
                    except (OSError, EOFError):
                        queue.insert(0, (tid, kw))
                        w.task = None
                    if progress:
                        progress(done, len(tasks), tid)
            if not queue and not any(w.task for w in pool):
                break
            ready = wait([w.conn for w in pool], timeout=0.5)
            for i, w in enumerate(pool):
                if w.conn in ready:
                    try:
                        status, tid, val = w.conn.recv()
                    except (EOFError, OSError):
                        status, tid, val = "crashed", w.task, None
                    if status == "ready":
                        w.ready = True
                        continue
                    if tid is not None:
                        el = time.time() - w.started
                        if status == "ok":
                            finish(tid, TaskResult("ok", val, "", el))
                        else:
                            finish(tid, TaskResult(status, None, CRASH_MSG if status == "crashed" else str(val), el))
                    w.task = None
                    if status == "crashed":
                        replace(i)
                    continue
                if w.task is not None:                          # watchdog: time and memory
                    el = time.time() - w.started
                    why = None
                    if timeout_s and el > timeout_s:
                        why = ("timeout", f"stopped after {el:.0f} s (time limit {timeout_s:.0f} s per model)")
                    elif w.rss_mb() > mem_limit:
                        why = ("memory", f"stopped: needed more than {mem_limit / 1024:.1f} GB of memory")
                    elif not w.alive():
                        why = ("crashed", CRASH_MSG)
                    if why:
                        finish(w.task, TaskResult(why[0], None, why[1], el))
                        w.task = None
                        replace(i)
                elif not w.alive():
                    replace(i)
    finally:
        for w in pool:
            try:
                w.send(None)
            except Exception:  # noqa: BLE001
                pass
            w.kill()
        try:
            os.remove(path)
        except OSError:
            pass
    return results


def _probe(payload, kind: str = "ok", seconds: float = 0.0, mb: int = 0):
    """Self-test task used by the test suite: behaves well, hangs, eats memory or crashes."""
    if kind == "sleep":
        time.sleep(seconds)
    elif kind == "memory":
        blocks = []
        for _ in range(mb // 50):
            blocks.append(bytearray(50 * 2 ** 20))
            time.sleep(0.05)
        time.sleep(5)
    elif kind == "crash":
        os._exit(9)
    elif kind == "raise":
        raise ValueError("boom")
    return {"kind": kind, "n": len(payload.get("data", []))}


def resolve_jobs(n: Optional[int]) -> int:
    """n_jobs value for scikit-learn: 0 / None / negative mean all available cores."""
    cpus = available_cpus()
    return cpus if not n or n < 1 else min(int(n), cpus)


class RunnerError(RuntimeError):
    """A protected task was stopped (time / memory) or failed; ``status`` says which."""

    def __init__(self, status: str, message: str):
        super().__init__(message)
        self.status = status


def run_one(payload: dict, fn: Callable, timeout_s: Optional[float] = None, mem_limit_mb: Optional[int] = None,
            **kwargs):
    """Run a single heavy task in an isolated worker and return its value (or raise RunnerError)."""
    mem = mem_limit_mb or int(available_memory_mb() * 0.8)
    r = run_isolated(payload, fn, [("task", kwargs)], workers=1, timeout_s=timeout_s, mem_limit_mb=mem)["task"]
    if r.status != "ok":
        raise RunnerError(r.status, r.message.split("\n")[0])
    return r.value
