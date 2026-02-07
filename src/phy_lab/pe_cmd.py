from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


class PEScriptError(RuntimeError):
    pass


_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,31}$")
_NODE_RE = re.compile(r"^[A-Za-z0-9_:+.-]{1,32}$")
_PIN_REF_RE = re.compile(r"^(?P<id>[A-Za-z][A-Za-z0-9_]{0,31})\.(?P<pin>[01])$")
_PARAM_KEY_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,31}$")
_ENG_NUM_RE = re.compile(
    r"^\s*(?P<num>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    r"(?P<suffix>meg|[pnumkKMGµ]?)"
    r"(?P<unit>[A-Za-zΩ]*)\s*$"
)


def _node_key(name: str) -> str:
    n = (name or "").strip()
    if not n:
        return ""
    low = n.casefold()
    if low in ("gnd", "ground", "0"):
        return "gnd"
    return n


def _require_id(value: str, *, where: str) -> str:
    s = (value or "").strip()
    if not _ID_RE.match(s):
        raise PEScriptError(f"{where} must match {_ID_RE.pattern} (got {value!r})")
    return s


def _require_node(value: str, *, where: str) -> str:
    n = _node_key(value)
    if not n:
        raise PEScriptError(f"{where} must be non-empty")
    if n != "gnd" and not _NODE_RE.match(n):
        raise PEScriptError(f"{where} invalid node name {value!r} (pattern={_NODE_RE.pattern})")
    return n


def _parse_float(token: str, *, where: str) -> float:
    token = (token or "").strip()
    if not token:
        raise PEScriptError(f"{where} must be a number (got {token!r})")

    try:
        v = float(token)
    except Exception:
        m = _ENG_NUM_RE.match(token)
        if not m:
            raise PEScriptError(f"{where} must be a number (got {token!r})")
        try:
            base = float(m.group("num"))
        except Exception as e:
            raise PEScriptError(f"{where} must be a number (got {token!r})") from e

        suf = (m.group("suffix") or "").strip()
        suf_low = suf.casefold()
        scale_map = {
            "p": 1e-12,
            "n": 1e-9,
            "u": 1e-6,
            "µ": 1e-6,
            "m": 1e-3,
            "k": 1e3,
            "K": 1e3,
            "M": 1e6,
            "G": 1e9,
        }
        scale = 1e6 if suf_low == "meg" else scale_map.get(suf, 1.0)
        v = base * float(scale)
    if v != v or v in (float("inf"), float("-inf")):
        raise PEScriptError(f"{where} must be finite (got {token!r})")
    return v


def _canonical_type(t: str) -> str:
    x = (t or "").strip().lower()
    if x in ("r", "resistor"):
        return "resistor"
    if x in ("c", "cap", "capacitor"):
        return "capacitor"
    if x in ("l", "ind", "inductor"):
        return "inductor"
    if x in ("vdc",):
        return "vdc"
    if x in ("idc",):
        return "idc"
    if x in ("vac",):
        return "vac"
    if x in ("iac",):
        return "iac"
    return x


def _param_key_map(ctype: str) -> dict[str, str]:
    # Maps DSL keys to pe_builder expected keys.
    if ctype == "resistor":
        return {"r": "r_ohm", "r_ohm": "r_ohm", "ohm": "r_ohm"}
    if ctype == "capacitor":
        return {"c": "c_f", "c_f": "c_f", "f": "c_f"}
    if ctype == "inductor":
        return {"l": "l_h", "l_h": "l_h", "h": "l_h"}
    if ctype == "vdc":
        return {"v": "v_v", "v_v": "v_v", "volt": "v_v"}
    if ctype == "idc":
        return {"i": "i_a", "i_a": "i_a", "amp": "i_a"}
    if ctype == "vac":
        return {
            "vp": "vp_v",
            "vp_v": "vp_v",
            "freq": "freq_hz",
            "freq_hz": "freq_hz",
            "phase": "phase_deg",
            "phase_deg": "phase_deg",
        }
    if ctype == "iac":
        return {
            "ip": "ip_a",
            "ip_a": "ip_a",
            "freq": "freq_hz",
            "freq_hz": "freq_hz",
            "phase": "phase_deg",
            "phase_deg": "phase_deg",
        }
    return {}


@dataclass
class _Comp:
    cid: str
    ctype: str
    n0: str
    n1: str
    params: dict[str, float]


def parse_pe_script_to_spec_obj(
    script: str,
    *,
    max_components: int = 30,
    max_probes: int = 20,
) -> dict[str, Any]:
    """Parse a safe, line-based circuit command script to a PE sim spec object.

    Supported commands (case-insensitive, one per line):
      - ANALYSIS <dc|ac|tr>
      - SET AC_OMEGA <omega_rad_s>
      - SET TR <t_step_s> <t_stop_s>
      - ADD <ID> <TYPE> <NODE0> <NODE1> <k=v ...>
      - WIRE <NODE> <ID.PIN> [ID.PIN ...]     (PIN is 0 or 1)
      - PROBE NODE <NODE>
      - PROBE I <ID>
      - PROBE VDROP <ID>
      - RUN                                  (optional; parser ignores but allows it)

    Notes:
      - Wiring is defined by node names: pins with the same node are connected.
      - `gnd` is the only special node name (ground).
    """
    text = (script or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")

    analysis_type: str | None = None
    ac_omega: float | None = None
    tr_step: float | None = None
    tr_stop: float | None = None

    comps: dict[str, _Comp] = {}
    probes: list[dict[str, str]] = []

    def add_probe(kind: str, target: str) -> None:
        if max_probes > 0 and len(probes) >= max_probes:
            raise PEScriptError(f"Too many probes (limit={max_probes})")
        probes.append({"kind": kind, "target": target})

    for lineno, raw in enumerate(lines, start=1):
        s = (raw or "").strip()
        if not s:
            continue
        if s.startswith("#") or s.startswith("//"):
            continue

        parts = s.split()
        cmd = parts[0].strip().upper()
        args = parts[1:]

        if cmd in ("ANALYSIS",):
            if len(args) != 1:
                raise PEScriptError(f"Line {lineno}: ANALYSIS expects 1 arg")
            at = args[0].strip().lower()
            if at not in ("dc", "ac", "tr"):
                raise PEScriptError(f"Line {lineno}: ANALYSIS must be dc|ac|tr (got {args[0]!r})")
            analysis_type = at
            continue

        if cmd in ("SET",):
            if not args:
                raise PEScriptError(f"Line {lineno}: SET expects a subcommand")
            sub = args[0].strip().upper()
            rest = args[1:]
            if sub in ("AC_OMEGA", "AC", "OMEGA"):
                if len(rest) != 1:
                    raise PEScriptError(f"Line {lineno}: SET AC_OMEGA expects 1 arg")
                ac_omega = _parse_float(rest[0], where=f"Line {lineno}: omega")
                continue
            if sub in ("TR",):
                if len(rest) != 2:
                    raise PEScriptError(f"Line {lineno}: SET TR expects 2 args")
                tr_step = _parse_float(rest[0], where=f"Line {lineno}: t_step")
                tr_stop = _parse_float(rest[1], where=f"Line {lineno}: t_stop")
                continue
            if sub in ("ANALYSIS",):
                if len(rest) != 1:
                    raise PEScriptError(f"Line {lineno}: SET ANALYSIS expects 1 arg")
                at = rest[0].strip().lower()
                if at not in ("dc", "ac", "tr"):
                    raise PEScriptError(f"Line {lineno}: SET ANALYSIS must be dc|ac|tr (got {rest[0]!r})")
                analysis_type = at
                continue
            raise PEScriptError(f"Line {lineno}: Unknown SET subcommand {sub!r}")

        if cmd in ("ADD",):
            if len(args) < 5:
                raise PEScriptError(f"Line {lineno}: ADD expects at least 5 args")
            cid = _require_id(args[0], where=f"Line {lineno}: component id")
            if cid in comps:
                raise PEScriptError(f"Line {lineno}: Duplicate component id {cid!r}")
            ctype = _canonical_type(args[1])
            n0 = _require_node(args[2], where=f"Line {lineno}: node0")
            n1 = _require_node(args[3], where=f"Line {lineno}: node1")

            key_map = _param_key_map(ctype)
            if not key_map:
                raise PEScriptError(f"Line {lineno}: Unsupported component type {args[1]!r}")

            params: dict[str, float] = {}
            for tok in args[4:]:
                if "=" not in tok:
                    raise PEScriptError(f"Line {lineno}: Param must be k=v (got {tok!r})")
                k, v = tok.split("=", 1)
                kk = (k or "").strip().lower()
                if not _PARAM_KEY_RE.match(kk):
                    raise PEScriptError(f"Line {lineno}: Invalid param key {k!r}")
                if kk not in key_map:
                    raise PEScriptError(f"Line {lineno}: Unsupported param {k!r} for type {ctype}")
                vv = _parse_float(v.strip(), where=f"Line {lineno}: {k}")
                params[key_map[kk]] = vv

            comps[cid] = _Comp(cid=cid, ctype=ctype, n0=n0, n1=n1, params=params)
            if max_components > 0 and len(comps) > max_components:
                raise PEScriptError(f"Too many components (limit={max_components})")
            continue

        if cmd in ("WIRE", "NET", "CONNECT"):
            if len(args) < 2:
                raise PEScriptError(f"Line {lineno}: WIRE expects at least 2 args")
            node = _require_node(args[0], where=f"Line {lineno}: node")
            for ref in args[1:]:
                m = _PIN_REF_RE.match(ref.strip())
                if not m:
                    raise PEScriptError(f"Line {lineno}: Invalid pin ref {ref!r} (expected ID.0 or ID.1)")
                cid = m.group("id")
                pin = int(m.group("pin"))
                comp = comps.get(cid)
                if comp is None:
                    raise PEScriptError(f"Line {lineno}: Unknown component {cid!r} in WIRE")
                if pin == 0:
                    comp.n0 = node
                else:
                    comp.n1 = node
            continue

        if cmd in ("PROBE",):
            if len(args) < 2:
                raise PEScriptError(f"Line {lineno}: PROBE expects at least 2 args")
            kind = args[0].strip().upper()
            target = " ".join(args[1:]).strip()
            if kind in ("NODE", "VNODE", "VN"):
                n = _require_node(target, where=f"Line {lineno}: node")
                add_probe("node_voltage", n)
                continue
            if kind in ("I", "CURRENT"):
                cid = _require_id(target, where=f"Line {lineno}: component id")
                add_probe("component_current", cid)
                continue
            if kind in ("VDROP", "V", "DROP"):
                cid = _require_id(target, where=f"Line {lineno}: component id")
                add_probe("component_vdrop", cid)
                continue
            raise PEScriptError(f"Line {lineno}: Unsupported PROBE kind {args[0]!r}")

        if cmd in ("RUN", "ANALYZE", "SIMULATE"):
            # Allowed no-op marker. Execution is handled by the caller.
            continue

        raise PEScriptError(f"Line {lineno}: Unknown command {cmd!r}")

    if analysis_type is None:
        analysis_type = "dc"

    components: list[dict[str, Any]] = []
    for cid in sorted(comps.keys()):
        c = comps[cid]
        components.append(
            {
                "id": c.cid,
                "type": c.ctype,
                "nodes": [c.n0, c.n1],
                "params": dict(c.params),
            }
        )

    return {
        "analysis": {
            "type": analysis_type,
            "ac_omega_rad_s": ac_omega,
            "tr_t_step_s": tr_step,
            "tr_t_stop_s": tr_stop,
        },
        "components": components,
        "probes": probes,
    }
