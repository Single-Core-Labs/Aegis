from __future__ import annotations

import contextlib
import time
from typing import Iterator


@contextlib.contextmanager
def timed() -> Iterator[None]:
    """Time a block with perf_counter_ns; yields nothing, exposes duration via a holder."""
    start = time.perf_counter_ns()
    holder: dict[str, float] = {"seconds": 0.0}
    try:
        yield holder
    finally:
        holder["seconds"] = (time.perf_counter_ns() - start) / 1e9


class Timer:
    """Reusable timer holder returned by start_timer()."""

    __slots__ = ("_start_ns", "seconds")

    def __init__(self) -> None:
        self._start_ns = time.perf_counter_ns()
        self.seconds = 0.0

    def stop(self) -> float:
        self.seconds = (time.perf_counter_ns() - self._start_ns) / 1e9
        return self.seconds