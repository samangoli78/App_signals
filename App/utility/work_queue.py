"""Reusable single-slot background worker with latest-wins semantics."""

from __future__ import annotations

import threading
import traceback
from typing import Any, Callable


class LatestWinsWorker:
    """Single dedicated worker thread, single pending-request slot."""

    def __init__(self, tk_after: Callable[[int, Callable[[], None]], Any], name: str = "bg-worker") -> None:
        self._tk_after = tk_after
        self._name = name
        self._cond = threading.Condition()
        self._pending: tuple | None = None
        self._stop = False
        self._serial = 0
        self._applied_serial = 0
        self._thread: threading.Thread | None = None

    def post(
        self,
        compute_fn: Callable[[Any], Any],
        apply_fn: Callable[[Any, Any], None],
        payload: Any,
    ) -> int:
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

    def _ensure_thread(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True, name=self._name)
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
