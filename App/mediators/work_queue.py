"""Reusable single-slot background worker with latest-wins semantics.

Drop this into any Tk-driven pipeline that wants to offload heavy CPU work
without freezing the UI. Each :meth:`LatestWinsWorker.post` registers a job
into a *single* pending slot — older un-served jobs are silently discarded so
a rapid burst of clicks / selections / slider drags never piles up work
behind the UI.

The compute callable runs on a daemon thread (released GIL during NumPy/SciPy
work, so the Tk main loop and the 3D viewer keep ticking). The apply callable
is scheduled back onto the Tk main loop via ``tk_after`` so any
matplotlib / Tk / GL operation it does stays on the main thread, where those
libraries actually expect to be touched.

Stale results are dropped on the Tk side too, so even if a job finished
"after" a newer one was posted, only the newest result reaches the UI.
"""
from __future__ import annotations

import threading
import traceback
from typing import Any, Callable


class LatestWinsWorker:
    """Single dedicated worker thread, single pending-request slot.

    Parameters
    ----------
    tk_after
        Bound ``after`` method (e.g. ``app.after`` or ``widget.after``). Used
        to schedule the apply callback back onto the Tk main loop.
    name
        Thread name, useful when debugging deadlocks in a profiler / py-spy.
    """

    def __init__(self, tk_after: Callable[[int, Callable[[], None]], Any], name: str = "bg-worker") -> None:
        self._tk_after = tk_after
        self._name = name
        self._cond = threading.Condition()
        self._pending: tuple | None = None
        self._stop = False
        self._serial = 0
        self._applied_serial = 0
        self._thread: threading.Thread | None = None

    # ----------------------------------------------------------- public API
    def post(
        self,
        compute_fn: Callable[[Any], Any],
        apply_fn: Callable[[Any, Any], None],
        payload: Any,
    ) -> int:
        """Register a new job; returns the serial assigned to it.

        ``compute_fn(payload)`` runs on the worker; its return value is passed
        to ``apply_fn(payload, result)`` on the Tk thread. Any older pending
        request is overwritten — only the latest payload survives.
        """
        self._ensure_thread()
        with self._cond:
            self._serial += 1
            self._pending = (compute_fn, apply_fn, payload, self._serial)
            self._cond.notify()
            return self._serial

    def is_busy(self) -> bool:
        return self._pending is not None or self._serial != self._applied_serial

    def stop(self) -> None:
        with self._cond:
            self._stop = True
            self._cond.notify_all()

    # ---------------------------------------------------------- internals
    def _ensure_thread(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop = False
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name=self._name
        )
        self._thread.start()

    def _loop(self) -> None:
        while True:
            with self._cond:
                while self._pending is None and not self._stop:
                    self._cond.wait()
                if self._stop:
                    return
                job = self._pending
                self._pending = None
                latest = self._serial
            if job is None:
                continue
            compute_fn, apply_fn, payload, serial = job
            if serial != latest:
                continue
            try:
                result = compute_fn(payload)
            except Exception:
                traceback.print_exc()
                continue

            def _go(p=payload, r=result, fn=apply_fn, sn=serial):
                # Drop the result if a newer request was posted (or applied)
                # in the meantime — keeps the UI on the freshest data only.
                if sn < self._applied_serial or sn != self._serial:
                    return
                self._applied_serial = sn
                try:
                    fn(p, r)
                except Exception:
                    traceback.print_exc()

            try:
                self._tk_after(0, _go)
            except Exception:
                traceback.print_exc()
