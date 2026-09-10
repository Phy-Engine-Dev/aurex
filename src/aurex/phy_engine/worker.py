"""Isolated native simulation worker; JSON on stdin/stdout, diagnostics on stderr."""
from __future__ import annotations

import json
import sys


def main() -> int:
    try:
        if sys.platform != "win32":
            import resource
            resource.setrlimit(resource.RLIMIT_CPU, (240, 245))
            resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
        payload = json.loads(sys.stdin.read(2_000_001))
        from aurex.tools.phy_engine import _simulate_spec
        from aurex.phy_engine.limits import DEFAULT_DIGITAL_COMPONENT_LIMIT
        result = _simulate_spec(payload["spec"], payload["lib_path"], return_state=bool(payload.get("return_state")),
                                digital_component_limit=payload.get("digital_component_limit", DEFAULT_DIGITAL_COMPONENT_LIMIT))
        print(json.dumps(result, allow_nan=False))
        return 0
    except Exception as error:
        # Preserve bounded machine-readable settle evidence across the worker
        # boundary. Prefixing/truncating stderr could otherwise remove the
        # failure reason or the exact conflicting drivers.
        try:
            diagnostic = json.loads(str(error))
        except (ValueError, TypeError):
            diagnostic = None
        if isinstance(diagnostic, dict) and diagnostic.get("execution_status") == "failed" and "digital_settle" in diagnostic:
            print(json.dumps({"error": diagnostic}, allow_nan=False))
            return 1
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
