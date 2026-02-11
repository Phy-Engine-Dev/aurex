from __future__ import annotations

import signal
from dataclasses import dataclass
from types import FrameType
from typing import Any, Callable


@dataclass
class GracefulShutdown:
    """Two-stage Ctrl+C handling.

    - First SIGINT: request a graceful stop (no new work, finish current reply).
    - Second SIGINT: force exit immediately.
    """

    logger: Any
    stop_requested: bool = False
    _count: int = 0
    _prev_handler: Callable[[int, FrameType | None], Any] | int | None = None

    def install(self) -> None:
        self._prev_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._on_sigint)

    def restore(self) -> None:
        if self._prev_handler is None:
            return
        signal.signal(signal.SIGINT, self._prev_handler)  # type: ignore[arg-type]
        self._prev_handler = None

    def _on_sigint(self, _signum: int, _frame: FrameType | None) -> None:
        self._count += 1
        if self._count == 1:
            self.stop_requested = True
            try:
                self.logger.warning("Ctrl+C: stop receiving new messages; finishing current reply. Press Ctrl+C again to force exit.")
            except Exception:
                pass
            return
        try:
            self.logger.error("Ctrl+C: force exit now.")
        except Exception:
            pass
        raise SystemExit(130)

