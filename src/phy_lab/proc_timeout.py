from __future__ import annotations

import multiprocessing as mp
import os
from dataclasses import dataclass
from typing import Any, Callable


class ProcTimeoutError(TimeoutError):
    pass


@dataclass(frozen=True)
class ProcRemoteError(RuntimeError):
    message: str
    type_name: str = "Exception"

    def __str__(self) -> str:
        return f"{self.type_name}: {self.message}"


def _worker(conn: Any, fn: Callable[..., Any], kwargs: dict[str, Any]) -> None:
    try:
        out = fn(**kwargs)
        conn.send(("ok", out))
    except BaseException as e:  # noqa: BLE001 - we need to capture anything from native bindings
        conn.send(("err", {"type": type(e).__name__, "msg": str(e)}))
    finally:
        try:
            conn.close()
        except Exception:
            pass


def run_with_timeout(
    *,
    fn: Callable[..., Any],
    kwargs: dict[str, Any],
    timeout_sec: float,
    label: str = "task",
) -> Any:
    if timeout_sec is None or float(timeout_sec) <= 0:
        return fn(**kwargs)

    ctx = mp.get_context("fork" if os.name == "posix" else mp.get_start_method())
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    p = ctx.Process(target=_worker, args=(child_conn, fn, kwargs), daemon=True)
    p.start()
    child_conn.close()

    try:
        if parent_conn.poll(float(timeout_sec)):
            tag, payload = parent_conn.recv()
            if tag == "ok":
                return payload
            if isinstance(payload, dict):
                raise ProcRemoteError(message=str(payload.get("msg") or ""), type_name=str(payload.get("type") or "Exception"))
            raise ProcRemoteError(message=str(payload), type_name="Exception")

        raise ProcTimeoutError(f"{label} timed out after {float(timeout_sec):g}s")
    finally:
        try:
            parent_conn.close()
        except Exception:
            pass
        if p.is_alive():
            p.terminate()
        p.join(timeout=1.0)

