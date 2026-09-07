"""Vendored Physics-Lab-AR publication transport.

This is the small publication-only subset of physicsLab 2.0.6
``physicsLab/_core.py`` and ``physicsLab/web/_api.py`` used by Aurex. Keeping
the official request construction here prevents the server path from drifting
away from the SDK while Aurex retains its durable one-shot publication ledger.

MIT License

Copyright (c) 2024 Arendelle

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

import gzip
import json
import os
import tempfile
from typing import Any

CURRENT_API_VERSION = "2503"


def _official_planet(model, identifier: str, position: str, parent: str | None) -> dict[str, Any]:
    """Read the model defaults directly from the installed official SDK class."""
    instance = model.__new__(model)
    model.__init__(instance, 0, 0, 0)
    value = dict(instance.data)
    value.update({"Identifier": identifier, "Parent": parent, "Position": position,
                  "Velocity": "0,0,0", "Acceleration": "0,0,0"})
    return value


def hdl_source_carrier_status() -> dict[str, Any]:
    """Fixed three-body carrier accepted by the current official API.

    Live API verification showed that zero- and one-element Type-3 workspaces
    are rejected with Publish.Failed.5. The canonical Sun/Earth/Moon shell is
    immutable and is never offered to the agent as an editable experiment.
    """
    from physicsLab.celestial import Earth, Moon, Sun  # type: ignore

    sun_id = "AUREX_HDL_CARRIER_SUN"
    earth_id = "AUREX_HDL_CARRIER_EARTH"
    moon_id = "AUREX_HDL_CARRIER_MOON"
    elements = {
        sun_id: _official_planet(Sun, sun_id, "0,0,0", None),
        earth_id: _official_planet(Earth, earth_id, "1,0,0", sun_id),
        moon_id: _official_planet(Moon, moon_id, "1.00257,0,0", earth_id),
    }
    return {"MainIdentifier": None, "Elements": elements,
        "WorldTime": 0.0, "ScalingName": "内太阳系", "LengthScale": 1.0,
        "SizeLinear": 0.0001, "SizeNonlinear": 0.5,
        "StarPresent": False, "Setting": None}


def _real(user: Any) -> Any:
    candidate = getattr(user, "user", user)
    if not all(isinstance(getattr(candidate, name, None), str) and getattr(candidate, name)
               for name in ("token", "auth_code")):
        raise PermissionError("official publication requires a bound Physics-Lab-AR user")
    return candidate


def _version() -> str:
    from physicsLab import plAR  # type: ignore

    value = plAR.get_plAR_version()
    return f"{value[0]}{value[1]}{value[2]}" if value is not None else CURRENT_API_VERSION


def submit_experiment(user: Any, payload: dict[str, Any]) -> dict[str, Any]:
    """Exact official SDK SubmitExperiment envelope (one request, no retry)."""
    from physicsLab.web import _request  # type: ignore
    from physicsLab.web._api import _check_response  # type: ignore

    real = _real(user)
    headers = {
        "x-API-Token": real.token,
        "x-API-AuthCode": real.auth_code,
        "x-API-Version": _version(),
        "Accept-Encoding": "gzip",
        "Content-Type": "gzipped/json",
    }
    # Match physicsLab/_core.py: default json.dumps settings are intentional.
    body = gzip.compress(json.dumps(payload).encode("utf-8"))
    response = _request.post_https(
        domain="physics-api-cn.turtlesim.com",
        path="Contents/SubmitExperiment",
        header=headers,
        body=body,
    )
    return _check_response(response)


def confirm_experiment(user: Any, summary_id: str, image_counter: int) -> dict[str, Any]:
    """Use the official SDK ConfirmExperiment implementation unchanged."""
    from physicsLab import Category  # type: ignore

    return _real(user).confirm_experiment(summary_id, Category.Experiment, image_counter)


def upload_image(user: Any, policy: str, authorization: str, cover: bytes) -> dict[str, Any]:
    """Use the official SDK UploadImage implementation with a private temp file."""
    if not isinstance(cover, bytes) or not cover:
        raise ValueError("cover must be non-empty bytes")
    fd, path = tempfile.mkstemp(prefix="aurex-official-cover-", suffix=".jpg")
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(cover)
            output.flush()
            os.fsync(output.fileno())
        return _real(user).upload_image(policy, authorization, path)
    finally:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
