from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from .errors import PLARError
from .http import _DEFAULT_TIMEOUT_SEC, configure_requests_default_timeout


def repo_root() -> str:
    """Return repository root (best-effort) based on the `src/` layout."""
    return str(Path(__file__).resolve().parents[2])


def unwrap_user(user: Any) -> Any:
    """Unwrap thin proxies that store the real user at `._user`."""
    inner = getattr(user, "_user", None)
    return inner if inner is not None else user


def _vendored_physicslab_dir() -> str:
    return os.path.join(repo_root(), "third-parties", "physicsLab")


def ensure_physicslab_importable(*, cache_dir: str, http_timeout_sec: float | None = None) -> None:
    """Use the repository's pinned PhysicsLab SDK and set its save directory.

    The SDK is vendored because the community API and PLSAV compatibility
    surface must not drift with an arbitrary PyPI installation.  A process
    which imported another SDK before Aurex starts is intentionally left alone
    rather than replacing live classes underneath it; normal Aurex startup
    reaches this function before importing ``physicsLab``.
    """
    cache_dir_abs = os.path.abspath(cache_dir)
    os.makedirs(cache_dir_abs, exist_ok=True)

    os.environ["PHYSICSLAB_HOME_PATH"] = os.path.join(cache_dir_abs, "physicsLabSav")
    configure_requests_default_timeout(_DEFAULT_TIMEOUT_SEC if http_timeout_sec is None else float(http_timeout_sec))

    vendored = _vendored_physicslab_dir()
    package = os.path.join(vendored, "physicsLab")
    if os.path.isdir(package) and vendored not in sys.path:
        # Put the repository copy ahead of site-packages.  This is deliberately
        # done before the import, not merely as a fallback after PyPI.
        sys.path.insert(0, vendored)

    try:
        import physicsLab  # noqa: F401
    except ImportError as e:
        raise PLARError(
            "The vendored `physicsLab` SDK is missing at "
            "`third-parties/physicsLab/physicsLab`. Restore the repository "
            "dependency instead of installing an unpinned PyPI SDK."
        ) from e
