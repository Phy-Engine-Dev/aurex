from __future__ import annotations

import json
import hashlib
import html
import gzip
import os
import re
import shutil
import tempfile
import time
import urllib.request
import urllib.parse
import uuid
from dataclasses import dataclass
from typing import Any

from ollama import OllamaClient
from plsav import PlSavError, load_plsav_counts
from pe_sim import AnalyzeType, PESimError, PhyEngineLib, SeriesVdcResistorsSpec
from pe_builder import (
    PEBuilderError,
    build_circuit,
    PEProbe,
    parse_pe_sim_spec,
    parse_spec_json,
)
from pe_cmd import PEScriptError, parse_pe_script_to_spec_obj
from pe_tool import evaluate_probes, run_and_sample
from plsav_sim import PlSavSimError, build_pe_circuit_input_from_status_save
from proc_timeout import ProcRemoteError, ProcTimeoutError, run_with_timeout
from phy_engine import (
    PhyEngineError,
    Verilog2PlSavOptions,
    ensure_phyengine_lib,
    ensure_verilog2plsav,
    verilog_to_plsav,
)
from plar import (
    PLARError,
    best_effort_extract_text,
    iter_text_fields,
    query_experiments,
    repo_root,
    upload_sav_as_experiment,
)
from text import extract_fenced_code, truncate


def _default_user_agent() -> str:
    # Keep this stable and boring to reduce blocks, but still realistic.
    return (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )


def _http_get_text(
    *,
    url: str,
    proxy: str = "",
    timeout_sec: float = 20.0,
    user_agent: str = "",
) -> str:
    """Fetch a URL and return text content.

    Uses stdlib urllib to avoid external dependencies. Supports HTTP(S) proxy URLs
    (e.g. 'http://127.0.0.1:7897').
    """
    url = (url or "").strip()
    if not url:
        raise RuntimeError("Empty URL")

    ua = (user_agent or "").strip() or _default_user_agent()
    headers = {
        "User-Agent": ua,
        # Avoid gzip to keep decoding simple. If a server still gzips, we handle it.
        "Accept-Encoding": "identity",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    req = urllib.request.Request(url, headers=headers, method="GET")

    handlers: list[Any] = []
    p = (proxy or "").strip()
    if p:
        if "://" not in p:
            p = "http://" + p
        if p.lower().startswith("socks"):
            raise RuntimeError(
                "SOCKS proxy is not supported by urllib. Use an HTTP proxy URL, or install and use requests[socks]."
            )
        handlers.append(urllib.request.ProxyHandler({"http": p, "https": p}))

    opener = urllib.request.build_opener(*handlers)
    with opener.open(req, timeout=float(timeout_sec)) as resp:
        raw = resp.read()
        enc = (resp.headers.get("Content-Encoding") or "").casefold()
        if "gzip" in enc:
            try:
                raw = gzip.decompress(raw)
            except Exception:
                pass
        charset = resp.headers.get_content_charset() or "utf-8"
        try:
            return raw.decode(charset, errors="replace")
        except LookupError:
            return raw.decode("utf-8", errors="replace")


def render_help(*, command_prefix: str) -> str:
    p = command_prefix or "!"
    return (
        "Supported interactions:\n"
        "- Mention the agent and ask naturally, e.g. '@aurex introduce this experiment'.\n"
        "- Optional command mode (if enabled in config):\n"
        f"  - {p}help\n"
        f"  - {p}chat <text>\n"
        f"  - {p}summarize <text>\n"
        f"  - {p}search <query>\n"
        f"  - {p}circuit <spec>  (optional publish; requires config enablement)\n"
        f"  - {p}simulate <spec>  (DC series VDC+2R demo)\n"
        f"  - {p}google <query>  (optional web search)\n"
    )


def llm_chat(*, ollama: OllamaClient, system_prompt: str, user_text: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]
    return ollama.chat(messages=messages)


def llm_summarize(*, ollama: OllamaClient, system_prompt: str, text: str) -> str:
    prompt = (
        "Summarize the following text.\n"
        "Requirements:\n"
        "- Preserve key facts and numbers.\n"
        "- Use concise bullet points.\n"
        "- Reply in the same language as the input.\n\n"
        f"{text}"
    )
    return llm_chat(ollama=ollama, system_prompt=system_prompt, user_text=prompt)


def llm_build_pe_sim_spec_json(
    *,
    ollama: OllamaClient,
    user_text: str,
    context_json: dict[str, Any] | None,
    max_components: int,
    max_probes: int,
) -> str:
    """Ask the LLM to produce a strict JSON spec for PE circuit simulation."""
    context_blob = ""
    if context_json is not None:
        context_blob = (
            "Context JSON (current page):\n"
            + json.dumps(context_json, ensure_ascii=False, indent=2)
            + "\n\n"
        )
    prompt = (
        "Convert the user's request into a strict JSON specification for a Phy-Engine circuit simulation.\n"
        "\n"
        "CRITICAL TYPE RULES (common failures):\n"
        "- The top-level output MUST be ONE JSON OBJECT (not an array).\n"
        "- `analysis` MUST be an object.\n"
        "- `components` MUST be a JSON ARRAY ([]) even if there is only 1 component.\n"
        "- `probes` MUST be a JSON ARRAY ([]). If none, use [] (not null).\n"
        "- `nodes` MUST be an array of exactly 2 strings.\n"
        "- `params` MUST be an object; ALL values MUST be numbers (no strings like \"1k\" / \"10u\").\n"
        "\n"
        "You MUST follow this JSON schema exactly (no extra keys):\n"
        "{\n"
        '  "analysis": {\n'
        '    "type": "dc" | "ac" | "tr",\n'
        '    "ac_omega_rad_s": number | null,\n'
        '    "tr_t_step_s": number | null,\n'
        '    "tr_t_stop_s": number | null\n'
        "  },\n"
        '  "components": [\n'
        "    {\n"
        '      "id": "R1",\n'
        '      "type": "resistor" | "capacitor" | "inductor" | "vdc" | "idc" | "vac" | "iac",\n'
        '      "nodes": ["n1", "n2"],\n'
        '      "params": { ...numbers... }\n'
        "    }\n"
        "  ],\n"
        '  "probes": [\n'
        "    {\"kind\":\"node_voltage\"|\"component_current\"|\"component_vdrop\",\"target\":\"...\"}\n"
        "  ]\n"
        "}\n"
        "\n"
        "Component param rules:\n"
        "- resistor: params MUST contain {\"r_ohm\": <number>}\n"
        "- capacitor: params MUST contain {\"c_f\": <number>}\n"
        "- inductor: params MUST contain {\"l_h\": <number>}\n"
        "- vdc: params MUST contain {\"v_v\": <number>}\n"
        "- idc: params MUST contain {\"i_a\": <number>}\n"
        "- vac: params MUST contain {\"vp_v\": <number>, \"freq_hz\": <number>, \"phase_deg\": <number>}\n"
        "- iac: params MUST contain {\"ip_a\": <number>, \"freq_hz\": <number>, \"phase_deg\": <number>}\n"
        "\n"
        "Hard constraints:\n"
        f"- 1 <= components.length <= {int(max_components)}\n"
        f"- 0 <= probes.length <= {int(max_probes)}\n"
        "- You MUST include a ground node named exactly \"gnd\" connected to the circuit.\n"
        "- Include at least ONE source component: vdc|idc|vac|iac.\n"
        "- Use simple node names like: gnd, n1, n2, vin, vout.\n"
        "\n"
        "Analysis selection rules:\n"
        "- Use \"dc\" for static results.\n"
        "- Use \"tr\" if the user asks about timing / charging / oscillation, or if capacitors/inductors are present.\n"
        "- Use \"ac\" only if the user asks about frequency response; if so, set ac_omega_rad_s.\n"
        "- For \"tr\": set BOTH tr_t_step_s and tr_t_stop_s (e.g. 1e-6 and 1e-3).\n"
        "\n"
        "Minimal example (format only):\n"
        "{\n"
        '  \"analysis\": {\"type\":\"dc\",\"ac_omega_rad_s\":null,\"tr_t_step_s\":null,\"tr_t_stop_s\":null},\n'
        '  \"components\": [\n'
        '    {\"id\":\"V1\",\"type\":\"vdc\",\"nodes\":[\"vin\",\"gnd\"],\"params\":{\"v_v\":5}},\n'
        '    {\"id\":\"R1\",\"type\":\"resistor\",\"nodes\":[\"vin\",\"gnd\"],\"params\":{\"r_ohm\":1000}}\n'
        "  ],\n"
        '  \"probes\": [{\"kind\":\"node_voltage\",\"target\":\"vin\"}]\n'
        "}\n"
        "\n"
        "Output rules:\n"
        "- Output ONLY valid JSON (no markdown/code fences, no explanations).\n"
        "- Do NOT include any extra keys.\n"
        "\n"
        + context_blob
        + "User request:\n"
        + user_text.strip()
    )
    return ollama.chat(
        messages=[
            {
                "role": "system",
                "content": "You are a careful circuit engineer. Output strict JSON only.",
            },
            {"role": "user", "content": prompt},
        ]
    ).strip()


def _extract_json_object_text(text: str) -> str:
    """Best-effort extract a JSON object substring from LLM output."""
    s = (text or "").strip()
    if not s:
        return s
    start = s.find("{")
    end = s.rfind("}")
    if start < 0 or end < 0 or end <= start:
        return s
    return s[start : end + 1].strip()


def llm_fix_pe_sim_spec_json(
    *,
    ollama: OllamaClient,
    user_text: str,
    context_json: dict[str, Any] | None,
    max_components: int,
    max_probes: int,
    prior_json: str,
    error_text: str,
) -> str:
    """Ask the LLM to fix a previously invalid JSON spec."""
    context_blob = ""
    if context_json is not None:
        context_blob = (
            "Context JSON (current page):\n"
            + json.dumps(context_json, ensure_ascii=False, indent=2)
            + "\n\n"
        )
    prompt = (
        "You previously produced a JSON specification for a Phy-Engine circuit simulation, but it failed validation.\n"
        "Fix the JSON so it passes the validator.\n"
        "\n"
        "CRITICAL TYPE RULES (common failures):\n"
        "- Output MUST be ONE JSON OBJECT.\n"
        "- `analysis` MUST be an object.\n"
        "- `components` MUST be a JSON ARRAY ([]), not an object.\n"
        "- `probes` MUST be a JSON ARRAY ([]), not null.\n"
        "- `nodes` MUST be an array of exactly 2 strings.\n"
        "- `params` MUST be an object; ALL values MUST be numbers (no strings like \"1k\").\n"
        "\n"
        "You MUST follow this JSON schema exactly:\n"
        "{\n"
        '  "analysis": {\n'
        '    "type": "dc" | "ac" | "tr",\n'
        '    "ac_omega_rad_s": number | null,\n'
        '    "tr_t_step_s": number | null,\n'
        '    "tr_t_stop_s": number | null\n'
        "  },\n"
        '  "components": [\n'
        "    {\n"
        '      "id": "R1",\n'
        '      "type": "resistor" | "capacitor" | "inductor" | "vdc" | "idc" | "vac" | "iac",\n'
        '      "nodes": ["n1", "n2"],\n'
        '      "params": { "r_ohm"/"c_f"/"l_h"/"v_v"/"i_a"/("vp_v","freq_hz","phase_deg") : number }\n'
        "    }\n"
        "  ],\n"
        '  "probes": [\n'
        "    {\"kind\":\"node_voltage\"|\"component_current\"|\"component_vdrop\",\"target\":\"...\"}\n"
        "  ]\n"
        "}\n"
        "\n"
        "Hard constraints:\n"
        f"- components.length MUST be between 1 and {int(max_components)}.\n"
        f"- probes.length MUST be between 0 and {int(max_probes)}.\n"
        "- Every component must be a 2-terminal element (nodes length exactly 2).\n"
        "- You MUST include a ground node named exactly 'gnd'.\n"
        "- Use simple node names like: gnd, n1, n2, vin, vout.\n"
        "- Use SI units (Ohm, Farad, Henry, Volt, Ampere).\n"
        "- Include at least ONE source component: vdc|idc|vac|iac.\n"
        "\n"
        "Output rules:\n"
        "- Output ONLY valid JSON (no markdown, no comments).\n"
        "- Do NOT include any extra keys.\n"
        "\n"
        + context_blob
        + "User request:\n"
        + user_text.strip()
        + "\n\n"
        + "Validator error:\n"
        + truncate(error_text or "", max_chars=1200)
        + "\n\n"
        + "Prior invalid JSON:\n"
        + truncate(prior_json or "", max_chars=3000)
    )
    return ollama.chat(
        messages=[
            {
                "role": "system",
                "content": "You are a careful circuit engineer. Output strict JSON only.",
            },
            {"role": "user", "content": prompt},
        ]
    ).strip()


def llm_build_pe_sim_script(
    *,
    ollama: OllamaClient,
    user_text: str,
    context_json: dict[str, Any] | None,
    max_components: int,
    max_probes: int,
) -> str:
    """Ask the LLM to produce a safe line-based command script for PE simulation."""
    context_blob = ""
    if context_json is not None:
        context_blob = (
            "Context JSON (current page):\n"
            + json.dumps(context_json, ensure_ascii=False, indent=2)
            + "\n\n"
        )
    prompt = (
        "Write a PE-SCRIPT (a safe, line-based command script) to build and simulate a circuit.\n"
        "You must follow the command grammar exactly. Output ONLY the script lines.\n"
        "\n"
        "Command reference (one per line; case-insensitive):\n"
        "- ANALYSIS <dc|ac|tr>\n"
        "- SET AC_OMEGA <omega_rad_s>           (only for ac)\n"
        "- SET TR <t_step_s> <t_stop_s>        (only for tr)\n"
        "- ADD <ID> <TYPE> <NODE0> <NODE1> <k=v ...>\n"
        "    TYPE: resistor|r, capacitor|c, inductor|l, vdc, idc, vac, iac\n"
        "    Params:\n"
        "      - resistor: r=<ohm>\n"
        "      - capacitor: c=<farad>\n"
        "      - inductor: l=<henry>\n"
        "      - vdc: v=<volt>\n"
        "      - idc: i=<amp>\n"
        "      - vac: vp=<volt_peak> freq=<hz> phase=<deg>\n"
        "      - iac: ip=<amp_peak> freq=<hz> phase=<deg>\n"
        "- WIRE <NODE> <ID.PIN> [ID.PIN ...]   (optional; PIN is 0 or 1)\n"
        "- PROBE NODE <NODE>\n"
        "- PROBE I <ID>\n"
        "- PROBE VDROP <ID>\n"
        "- RUN                                 (optional marker)\n"
        "\n"
        "Wiring rules:\n"
        "- Pins that share the same NODE name are connected.\n"
        "- You MUST include a ground node named exactly 'gnd' connected to the circuit.\n"
        "\n"
        "Hard constraints:\n"
        f"- At most {int(max_components)} ADD commands.\n"
        f"- At most {int(max_probes)} PROBE commands.\n"
        "- Every component must be a 2-terminal element.\n"
        "- Use simple node names like: gnd, n1, n2, vin, vout.\n"
        "- Use SI units (Ohm, Farad, Henry, Volt, Ampere).\n"
        "\n"
        "Output rules:\n"
        "- Output ONLY script lines (no JSON, no markdown, no explanations).\n"
        "- Do NOT use semicolons.\n"
        "- No extra commands outside the reference.\n"
        "\n"
        + context_blob
        + "User request:\n"
        + user_text.strip()
    )
    return ollama.chat(
        messages=[
            {
                "role": "system",
                "content": "You are a careful circuit engineer. Output PE-SCRIPT only.",
            },
            {"role": "user", "content": prompt},
        ]
    ).strip()


def simulate_ai_circuit_with_phyengine(
    *,
    ollama: OllamaClient,
    text: str,
    context_json: dict[str, Any] | None,
    phy_engine_cfg: Any,
    config_base_dir: str,
    max_components: int = 30,
    max_probes: int = 20,
    max_attempts: int = 3,
) -> str:
    lang_zh = _looks_like_zh(text)
    if max_attempts <= 0:
        max_attempts = 1

    cmake_source_dir = _resolve_existing_path(
        str(getattr(phy_engine_cfg, "cmake_source_dir", "")),
        base_dir=config_base_dir,
    )
    lib_cfg_path = _resolve_existing_path(
        str(getattr(phy_engine_cfg, "phyengine_lib_path", "")),
        base_dir=config_base_dir,
    )
    lib_path = ensure_phyengine_lib(
        phyengine_lib_path=lib_cfg_path,
        auto_build=bool(getattr(phy_engine_cfg, "auto_build", False)),
        cmake_source_dir=cmake_source_dir,
        cmake_build_dir=os.path.abspath(
            os.path.join(config_base_dir, str(getattr(phy_engine_cfg, "cmake_build_dir", "")))
        ),
        cmake_build_type=str(getattr(phy_engine_cfg, "cmake_build_type", "Release")),
        build_timeout_sec=int(getattr(phy_engine_cfg, "build_timeout_sec", 900)),
    )

    raw_json = _extract_json_object_text(
        llm_build_pe_sim_spec_json(
            ollama=ollama,
            user_text=text,
            context_json=context_json,
            max_components=max_components,
            max_probes=max_probes,
        )
    )
    last_err: PEBuilderError | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            obj = parse_spec_json(raw_json)
            spec = parse_pe_sim_spec(obj, max_components=max_components, max_probes=max_probes)
            built = build_circuit(spec)
            last_err = None
            break
        except PEBuilderError as e:
            last_err = e
            if attempt >= max_attempts:
                break
            raw_json = _extract_json_object_text(
                llm_fix_pe_sim_spec_json(
                    ollama=ollama,
                    user_text=text,
                    context_json=context_json,
                    max_components=max_components,
                    max_probes=max_probes,
                    prior_json=raw_json,
                    error_text=str(e),
                )
            )

    if last_err is not None:
        return (
            "I couldn't build a valid circuit spec from your request. "
            f"Error: {last_err}"
            if not lang_zh
            else f"我没能从你的描述里构建出可仿真的电路规格。错误：{last_err}"
        )

    timeout_sec = _pe_sim_timeout_sec(phy_engine_cfg, default_sec=5.0)
    try:
        sample = run_and_sample(
            lib_path=lib_path,
            spec=spec,
            built=built,
            max_pins_per_comp=16,
            max_branches_per_comp=8,
            timeout_sec=timeout_sec,
        )
    except Exception as e:
        msg = str(e)
        return (
            f"Phy-Engine simulation failed: {msg}"
            if not lang_zh
            else f"Phy-Engine 仿真失败：{msg}"
        )

    comp_size = int(sample.comp_size)
    voltage = list(sample.voltage)
    voltage_ord = list(sample.voltage_ord)
    current = list(sample.current)
    current_ord = list(sample.current_ord)

    # Helpers for extracting values.
    def comp_index_for_element_index(element_index: int) -> int | None:
        if element_index <= 0:
            return None
        # Component order is non-ground elements in ascending element index.
        return element_index - 1

    def node_voltage(node: str) -> float | None:
        n = (node or "").strip()
        if not n:
            return None
        if n.casefold() in ("gnd", "ground", "0"):
            return 0.0
        ref = built.node_to_pin.get(n) or built.node_to_pin.get(n.casefold())  # best effort
        if not ref:
            return None
        ei, pin = ref
        ci = comp_index_for_element_index(ei)
        if ci is None or ci + 1 >= len(voltage_ord):
            return None
        start = voltage_ord[ci]
        end = voltage_ord[ci + 1]
        if (end - start) <= pin:
            return None
        return float(voltage[start + pin])

    def component_current(cid: str) -> float | None:
        ei = built.element_index_by_id.get(cid)
        if ei is None:
            return None
        ci = comp_index_for_element_index(ei)
        if ci is None or ci + 1 >= len(current_ord):
            return None
        start = current_ord[ci]
        end = current_ord[ci + 1]
        if end <= start:
            return None
        return float(current[start])

    def component_vdrop(cid: str) -> float | None:
        ei = built.element_index_by_id.get(cid)
        if ei is None:
            return None
        ci = comp_index_for_element_index(ei)
        if ci is None or ci + 1 >= len(voltage_ord):
            return None
        start = voltage_ord[ci]
        end = voltage_ord[ci + 1]
        if (end - start) < 2:
            return None
        v0 = float(voltage[start + 0])
        v1 = float(voltage[start + 1])
        return v0 - v1

    probes = list(spec.probes)
    if not probes:
        # Default probes: show node voltages (excluding gnd) and per-component current/vdrop (limited).
        node_names = [n for n in built.node_to_pin.keys() if n != "gnd"]
        for n in node_names[: max(0, max_probes // 2)]:
            probes.append(PEProbe(kind="node_voltage", target=n))
        for comp in spec.components[: max(0, max_probes // 2)]:
            probes.append(PEProbe(kind="component_current", target=comp.id))

    lines: list[str] = []
    if lang_zh:
        lines.append("仿真结果（Phy-Engine）:")
        lines.append(f"- analysis={spec.analysis_type}, components={len(spec.components)}")
    else:
        lines.append("Simulation result (Phy-Engine):")
        lines.append(f"- analysis={spec.analysis_type}, components={len(spec.components)}")

    for p in probes[:max_probes]:
        if p.kind == "node_voltage":
            v = node_voltage(p.target)
            if v is None:
                lines.append(f"- V({p.target}) = <unavailable>")
            else:
                lines.append(f"- V({p.target}) = {v:.6g} V")
        elif p.kind == "component_current":
            i = component_current(p.target)
            if i is None:
                lines.append(f"- I({p.target}) = <unavailable>")
            else:
                lines.append(f"- I({p.target}) = {i:.6g} A")
        else:
            vd = component_vdrop(p.target)
            if vd is None:
                lines.append(f"- Vdrop({p.target}) = <unavailable>")
            else:
                lines.append(f"- Vdrop({p.target}) = {vd:.6g} V")

    return "\n".join(lines)


def simulate_ai_script_circuit_with_phyengine(
    *,
    ollama: OllamaClient,
    text: str,
    context_json: dict[str, Any] | None,
    phy_engine_cfg: Any,
    config_base_dir: str,
    max_components: int = 30,
    max_probes: int = 20,
) -> str:
    """LLM -> PE-SCRIPT -> (parsed) -> Phy-Engine -> results -> LLM answer."""
    lang_zh = _looks_like_zh(text)

    cmake_source_dir = _resolve_existing_path(
        str(getattr(phy_engine_cfg, "cmake_source_dir", "")),
        base_dir=config_base_dir,
    )
    lib_cfg_path = _resolve_existing_path(
        str(getattr(phy_engine_cfg, "phyengine_lib_path", "")),
        base_dir=config_base_dir,
    )
    lib_path = ensure_phyengine_lib(
        phyengine_lib_path=lib_cfg_path,
        auto_build=bool(getattr(phy_engine_cfg, "auto_build", False)),
        cmake_source_dir=cmake_source_dir,
        cmake_build_dir=os.path.abspath(
            os.path.join(config_base_dir, str(getattr(phy_engine_cfg, "cmake_build_dir", "")))
        ),
        cmake_build_type=str(getattr(phy_engine_cfg, "cmake_build_type", "Release")),
        build_timeout_sec=int(getattr(phy_engine_cfg, "build_timeout_sec", 900)),
    )

    script = llm_build_pe_sim_script(
        ollama=ollama,
        user_text=text,
        context_json=context_json,
        max_components=max_components,
        max_probes=max_probes,
    )
    try:
        spec_obj = parse_pe_script_to_spec_obj(
            script,
            max_components=max_components,
            max_probes=max_probes,
        )
        spec = parse_pe_sim_spec(spec_obj, max_components=max_components, max_probes=max_probes)
        built = build_circuit(spec)
    except (PEScriptError, PEBuilderError) as e:
        return (
            f"我没能把命令脚本解析成可仿真的电路。错误：{e}\n\n脚本：\n{script}"
            if lang_zh
            else f"I couldn't parse the command script into a simulatable circuit. Error: {e}\n\nScript:\n{script}"
        )

    probes = list(spec.probes)
    if not probes:
        node_names = [n for n in built.node_to_pin.keys() if n != "gnd"]
        for n in node_names[: max(0, max_probes // 2)]:
            probes.append(PEProbe(kind="node_voltage", target=n))
        for comp in spec.components[: max(0, max_probes // 2)]:
            probes.append(PEProbe(kind="component_current", target=comp.id))

    timeout_sec = _pe_sim_timeout_sec(phy_engine_cfg, default_sec=5.0)
    try:
        sample = run_and_sample(lib_path=lib_path, spec=spec, built=built, timeout_sec=timeout_sec)
    except Exception as e:
        msg = str(e)
        return (
            f"Phy-Engine 仿真失败：{msg}"
            if lang_zh
            else f"Phy-Engine simulation failed: {msg}"
        )
    evaluated = evaluate_probes(
        built=built,
        sample=sample,
        probes=[{"kind": p.kind, "target": p.target} for p in probes[:max_probes]],
    )

    raw = {
        "analysis": {
            "type": spec.analysis_type,
            "ac_omega_rad_s": spec.ac_omega_rad_s,
            "tr_t_step_s": spec.tr_t_step_s,
            "tr_t_stop_s": spec.tr_t_stop_s,
        },
        "components": [
            {"id": c.id, "type": c.type, "nodes": list(c.nodes), "params": dict(c.params)}
            for c in spec.components
        ],
        "probes": evaluated,
    }

    messages: list[dict[str, str]] = []
    messages.append(
        {
            "role": "system",
            "content": (
                "Reply in Chinese. Use the simulation results precisely. Be concise."
                if lang_zh
                else "Reply in English. Use the simulation results precisely. Be concise."
            ),
        }
    )
    messages.append({"role": "system", "content": "PE-SCRIPT used:\n" + script.strip()})
    messages.append({"role": "system", "content": "Raw simulation results (JSON):\n" + json.dumps(raw, ensure_ascii=False)})
    messages.append(
        {
            "role": "user",
            "content": (
                "Use the simulation results above to answer the user's request.\n"
                "- If a requested value is missing/unavailable, say so briefly.\n"
                "- Include the most relevant numeric results.\n\n"
                f"User request:\n{text.strip()}"
            ),
        }
    )
    return ollama.chat(messages=messages).strip()


def search_recent_experiments(
    *,
    user: Any,
    query: str,
    max_scan: int = 200,
    max_results: int = 5,
) -> list[dict[str, Any]]:
    query = (query or "").strip()
    if not query:
        return []

    try:
        from physicsLab import Category  # type: ignore

        categories: list[Any] = [Category.Experiment, Category.Discussion]
    except Exception:  # pragma: no cover
        # Allow direct HTTP mode without the physicsLab dependency.
        categories = ["Experiment", "Discussion"]

    def _parse_hot_query(q: str) -> tuple[list[str], int] | None:
        q = (q or "").strip()
        if not q:
            return None
        qlow = q.casefold()
        if not (q.startswith("热门") or any(w in qlow for w in ("hot", "popular", "popularity"))):
            return None

        # Days: default 14 for "hot" queries if unspecified.
        days = 14
        m = re.search(r"(?i)\bdays?\s*=\s*(?P<n>[0-9]{1,3})\b", q) or re.search(
            r"(?i)\b(?P<n>[0-9]{1,3})\s*(?:d|day|days)\b", q
        ) or re.search(r"(?P<n>[0-9]{1,3})\s*天", q)
        if m and m.group("n"):
            try:
                days = int(m.group("n"))
            except Exception:
                days = 14
        if days < 0:
            days = 0
        if days > 365:
            days = 365

        cats: list[str] = []
        mc = re.search(r"(?i)\bcategory\s*=\s*(?P<cats>[A-Za-z/,_]+)\b", q)
        if mc and mc.group("cats"):
            raw = mc.group("cats")
            for part in re.split(r"[/,]", raw):
                p = (part or "").strip()
                if not p:
                    continue
                if p.casefold() in ("user", "experiment", "discussion", "model"):
                    cats.append(p[:1].upper() + p[1:].casefold())
        else:
            if ("用户" in q) or ("user" in qlow):
                cats.append("User")
            if ("实验" in q) or ("experiment" in qlow):
                cats.append("Experiment")
            if ("讨论" in q) or ("discussion" in qlow):
                cats.append("Discussion")

        if not cats:
            cats = ["Experiment", "Discussion"]

        # De-dup and keep stable order.
        cats = list(dict.fromkeys(cats))
        return cats, days

    hot = _parse_hot_query(query)
    if hot is not None:
        cats, days = hot
        out: list[dict[str, Any]] = []
        for c in cats:
            try:
                page = query_experiments(
                    user,
                    category=c,
                    take=min(24, max(1, int(max_results))),
                    skip=0,
                    from_skip=None,
                    days=days,
                    sort="Popularity",
                )
            except Exception:
                continue
            for item in page:
                if not isinstance(item, dict):
                    continue
                if "Category" not in item:
                    item = dict(item)
                    item["Category"] = c
                out.append(item)
                if len(out) >= max_results:
                    return out
        return out

    scanned: list[dict[str, Any]] = []

    for cat in categories:
        from_skip: str | None = None
        skip = 0
        while len(scanned) < max_scan:
            page_take = min(24, max_scan - len(scanned))
            page = query_experiments(
                user,
                category=cat,
                take=page_take,
                skip=skip,
                from_skip=from_skip,
            )
            if not page:
                break
            scanned.extend(page)
            skip += int(page_take)
            last = page[-1]
            last_id = best_effort_extract_text(last.get("ID")) or best_effort_extract_text(last.get("Id"))
            from_skip = last_id or from_skip
            if len(page) < page_take:
                break

    q = query.casefold()
    hits: list[dict[str, Any]] = []
    for item in scanned:
        hay = "\n".join(
            iter_text_fields(item, ["Subject", "Description", "Title", "Summary"])
        ).casefold()
        if q in hay:
            hits.append(item)
            if len(hits) >= max_results:
                break

    return hits


def format_experiment_hits(hits: list[dict[str, Any]]) -> str:
    if not hits:
        return "No results found (best-effort search over recent items only)."

    lines: list[str] = ["Top results:"]
    for idx, item in enumerate(hits, start=1):
        subject = best_effort_extract_text(item.get("Subject")) or "(no subject)"
        item_id = best_effort_extract_text(item.get("ID")) or best_effort_extract_text(
            item.get("Id")
        )
        cat = best_effort_extract_text(item.get("Category"))
        if item_id:
            if cat:
                lines.append(f"{idx}. [{cat}] {subject} (ID: {item_id})")
            else:
                lines.append(f"{idx}. {subject} (ID: {item_id})")
        else:
            if cat:
                lines.append(f"{idx}. [{cat}] {subject}")
            else:
                lines.append(f"{idx}. {subject}")
    return "\n".join(lines)


def llm_generate_verilog(*, ollama: OllamaClient, spec: str) -> str:
    prompt = (
        "Generate synthesizable Verilog-2001 (IEEE 1364-2001) code for the following specification.\n"
        "The output must be compatible with Phy-Engine 'verilog2plsav' (a strict synthesizable subset).\n"
        "\n"
        "Hard compatibility constraints (very important):\n"
        "- Verilog-2001 ONLY (no SystemVerilog).\n"
        "- The top module MUST be named 'top'.\n"
        "- Prefer a single module; if you use submodules, keep everything in one file.\n"
        "- Do NOT use: logic, always_comb, always_ff, always_latch, enum, struct, typedef, interface, package, class.\n"
        "- Do NOT use: inout ports, tri-states, multiple drivers on one net, wand/wor/tri, force/release.\n"
        "- Do NOT use: initial blocks, delays (#), wait/fork/join, system tasks ($display/$monitor), file I/O.\n"
        "- Do NOT use: memories/arrays (reg [..] mem [..]), multi-dimensional arrays, unpacked arrays.\n"
        "- Avoid complex generate/for/while; if absolutely necessary, use ONLY constant-bounded unrolling.\n"
        "- Keep arithmetic simple (prefer +, -, &, |, ^, ~, <<, >> with constant shifts). Avoid *, /, %, and variable shifts.\n"
        "\n"
        "Style constraints (recommended):\n"
        "- Use 'wire' for combinational nets and 'reg' for always blocks.\n"
        "- Combinational: use continuous 'assign' or 'always @(*)' with blocking '='.\n"
        "- Sequential: use 'always @(posedge clk)' with nonblocking '<='.\n"
        "- Keep ports and signals 1-D vectors only (e.g. [7:0]); avoid fancy packing.\n"
        "\n"
        "Output rules:\n"
        "- Output ONLY a single fenced code block.\n"
        "- The fenced block language tag must be 'verilog'.\n"
        "- Keep it minimal and correct.\n\n"
        f"Specification:\n{spec}\n"
    )
    response = ollama.chat(
        messages=[
            {
                "role": "system",
                "content": "You are a careful Verilog engineer. Follow the output rules strictly.",
            },
            {"role": "user", "content": prompt},
        ]
    )
    code = extract_fenced_code(response, preferred_lang="verilog") or response.strip()
    return code.strip()


def llm_fix_verilog(
    *,
    ollama: OllamaClient,
    spec: str,
    prior_verilog: str,
    error_text: str,
) -> str:
    prompt = (
        "The following Verilog failed to compile/export to a Physics Lab .sav.\n"
        "Fix the Verilog while preserving the original specification.\n\n"
        "Hard compatibility constraints (very important):\n"
        "- Verilog-2001 ONLY (no SystemVerilog).\n"
        "- The top module MUST be named 'top'.\n"
        "- Do NOT use: logic, always_comb, always_ff, always_latch, enum, struct, typedef, interface, package, class.\n"
        "- Do NOT use: inout ports, tri-states, multiple drivers on one net, wand/wor/tri, force/release.\n"
        "- Do NOT use: initial blocks, delays (#), wait/fork/join, system tasks ($display/$monitor), file I/O.\n"
        "- Do NOT use: memories/arrays (reg [..] mem [..]), multi-dimensional arrays, unpacked arrays.\n"
        "- Avoid complex generate/for/while; if absolutely necessary, use ONLY constant-bounded unrolling.\n"
        "- Keep arithmetic simple (prefer +, -, &, |, ^, ~, <<, >> with constant shifts). Avoid *, /, %, and variable shifts.\n\n"
        "Output rules:\n"
        "- Output ONLY a single fenced code block.\n"
        "- The fenced block language tag must be 'verilog'.\n"
        "- Keep it synthesizable Verilog-2001.\n\n"
        "Specification:\n"
        f"{spec}\n\n"
        "Compiler error:\n"
        f"{truncate(error_text, max_chars=3000)}\n\n"
        "Prior Verilog:\n"
        f"{prior_verilog}\n"
    )
    response = ollama.chat(
        messages=[
            {
                "role": "system",
                "content": "You are a careful Verilog engineer. Follow the output rules strictly.",
            },
            {"role": "user", "content": prompt},
        ]
    )
    code = extract_fenced_code(response, preferred_lang="verilog") or response.strip()
    return code.strip()


def _resolve_existing_path(path: str, *, base_dir: str) -> str:
    if not path:
        return ""
    if os.path.isabs(path):
        return path

    cand1 = os.path.abspath(os.path.join(base_dir, path))
    if os.path.exists(cand1):
        return cand1

    cand2 = os.path.abspath(os.path.join(repo_root(), path))
    if os.path.exists(cand2):
        return cand2

    return cand1


_RES_RE = re.compile(
    r"(?P<value>[0-9]+(?:\.[0-9]+)?)\s*(?P<scale>[kKmM]?)\s*(?:ohm|Ω|欧姆|欧)\b"
)
_VOLT_RE = re.compile(r"(?P<value>[0-9]+(?:\.[0-9]+)?)\s*(?:v|伏)\b", re.IGNORECASE)
_KV_RE = re.compile(r"(?i)\bV\s*=\s*(?P<value>[0-9]+(?:\.[0-9]+)?)\b")
_KR_RE = re.compile(r"(?i)\bR(?P<idx>[12])\s*=\s*(?P<value>[0-9]+(?:\.[0-9]+)?)\b")


def _parse_series_vdc_2r_spec(text: str) -> SeriesVdcResistorsSpec | None:
    text = (text or "").strip()
    if not text:
        return None

    v = None
    m = _KV_RE.search(text) or _VOLT_RE.search(text)
    if m:
        v = float(m.group("value"))

    r_matches = list(_RES_RE.finditer(text))
    r_values: list[float] = []
    for rm in r_matches:
        raw = float(rm.group("value"))
        scale = (rm.group("scale") or "").lower()
        mult = 1.0
        if scale == "k":
            mult = 1_000.0
        elif scale == "m":
            mult = 1_000_000.0
        r_values.append(raw * mult)

    # Allow "R1=100 R2=200" without unit (ohm assumed).
    r_kv = {m.group("idx"): float(m.group("value")) for m in _KR_RE.finditer(text)}
    if "1" in r_kv:
        r_values.insert(0, r_kv["1"])
    if "2" in r_kv:
        r_values.insert(1 if len(r_values) >= 1 else 0, r_kv["2"])

    if v is None or len(r_values) < 2:
        return None

    return SeriesVdcResistorsSpec(v_volts=v, r1_ohm=r_values[0], r2_ohm=r_values[1])


_N_SAME_RES_RE = re.compile(
    r"(?P<count>[0-9]{1,3})\s*(?:x|×|个)\s*(?P<value>[0-9]+(?:\.[0-9]+)?)\s*(?P<scale>[kKmM]?)\s*(?:ohm|Ω|欧姆|欧)\s*(?:电阻|resistors?)?",
    re.IGNORECASE,
)
_N_RES_ONLY_RE = re.compile(r"(?P<count>[0-9]{1,3})\s*(?:个)?\s*(?:电阻|resistors?)", re.IGNORECASE)


def _parse_series_vdc_n_resistors(text: str) -> tuple[float | None, list[float]]:
    """Best-effort parse for series VDC + N resistors."""
    text = (text or "").strip()
    if not text:
        return None, []

    v = None
    m = _KV_RE.search(text) or _VOLT_RE.search(text)
    if m:
        v = float(m.group("value"))

    # Pattern: "3x4ohm resistors"
    m2 = _N_SAME_RES_RE.search(text)
    if m2:
        count = int(m2.group("count"))
        raw = float(m2.group("value"))
        scale = (m2.group("scale") or "").lower()
        mult = 1.0
        if scale == "k":
            mult = 1_000.0
        elif scale == "m":
            mult = 1_000_000.0
        r = raw * mult
        if count > 0:
            return v, [r] * count

    r_matches = list(_RES_RE.finditer(text))
    r_values: list[float] = []
    for rm in r_matches:
        raw = float(rm.group("value"))
        scale = (rm.group("scale") or "").lower()
        mult = 1.0
        if scale == "k":
            mult = 1_000.0
        elif scale == "m":
            mult = 1_000_000.0
        r_values.append(raw * mult)

    # If only one R specified but count is mentioned, replicate.
    if len(r_values) == 1:
        m3 = _N_RES_ONLY_RE.search(text)
        if m3:
            count = int(m3.group("count"))
            if count > 1:
                r_values = [r_values[0]] * count

    return v, r_values


def simulate_series_vdc_resistors(
    *,
    text: str,
    phy_engine_cfg: Any,
    config_base_dir: str,
) -> str:
    lang_zh = _looks_like_zh(text)
    v, rs = _parse_series_vdc_n_resistors(text)
    if not rs:
        return (
            "To simulate a series circuit, include the number of resistors and their resistance.\n"
            "Example: 'simulate V=5V 3x4ohm resistors in series with a VDC'."
            if not lang_zh
            else "要进行串联直流仿真，请写清电阻数量与阻值。\n例如：'仿真 V=5V 3个4Ω电阻 串联 VDC'。"
        )
    if v is None:
        return (
            "I can simulate it, but I need the VDC voltage value. Example: 'V=5V'."
            if not lang_zh
            else "我可以仿真，但需要你给出 VDC 的电压值，例如：'V=5V'。"
        )

    cmake_source_dir = _resolve_existing_path(
        str(getattr(phy_engine_cfg, "cmake_source_dir", "")),
        base_dir=config_base_dir,
    )
    cmake_build_dir = os.path.abspath(
        os.path.join(config_base_dir, str(getattr(phy_engine_cfg, "cmake_build_dir", "")))
    )
    lib_cfg_path = _resolve_existing_path(
        str(getattr(phy_engine_cfg, "phyengine_lib_path", "")),
        base_dir=config_base_dir,
    )
    lib_path = ensure_phyengine_lib(
        phyengine_lib_path=lib_cfg_path,
        auto_build=bool(getattr(phy_engine_cfg, "auto_build", False)),
        cmake_source_dir=cmake_source_dir,
        cmake_build_dir=cmake_build_dir,
        cmake_build_type=str(getattr(phy_engine_cfg, "cmake_build_type", "Release")),
        build_timeout_sec=int(getattr(phy_engine_cfg, "build_timeout_sec", 900)),
    )

    pe = PhyEngineLib(lib_path)

    # elements: 0=ground placeholder, 1=VDC, 2..=resistors
    n = len(rs)
    element_codes = [0, 4] + [1] * n
    properties = [float(v)] + [float(r) for r in rs]

    wires: list[int] = []
    if n >= 1:
        # VDC(+) -> R1
        wires.extend([1, 0, 2, 0])
        # Chain resistors
        for i in range(0, n - 1):
            a = 2 + i
            b = 2 + i + 1
            wires.extend([a, 1, b, 0])
        # Rn -> VDC(-)
        wires.extend([2 + (n - 1), 1, 1, 1])
    # Tie VDC(-) to ground
    wires.extend([1, 1, 0, 0])

    circuit, vec_pos, chunk_pos, comp_size = pe.create_circuit(
        element_codes=element_codes,
        wires=wires,
        properties=properties,
    )
    try:
        pe.set_analyze_type(circuit=circuit, analyze_type=1)
        pe.analyze(circuit=circuit)
        voltage, voltage_ord, current, current_ord, _digital, _digital_ord = pe.sample(
            circuit=circuit,
            vec_pos=vec_pos,
            chunk_pos=chunk_pos,
            comp_size=comp_size,
            max_pins_per_comp=8,
            max_branches_per_comp=4,
        )
    finally:
        pe.destroy_circuit(circuit=circuit, vec_pos=vec_pos, chunk_pos=chunk_pos)

    # comp order: VDC(0), R1(1), R2(2)...
    i_r1 = 0.0
    if comp_size >= 2 and (current_ord[2] - current_ord[1]) >= 1:
        i_r1 = float(current[current_ord[1] + 0])
    i = abs(i_r1) if i_r1 != 0.0 else abs(float(v) / sum(rs))
    r_total = float(sum(rs))
    i_ma = i * 1000.0

    lines: list[str] = []
    if lang_zh:
        lines.append("直流仿真（VDC + 多个电阻串联）:")
        lines.append(f"- R_total = {r_total:.6g} Ω")
        lines.append(f"- I ≈ {i:.6g} A ({i_ma:.6g} mA)")
    else:
        lines.append("DC simulation (VDC + N resistors in series):")
        lines.append(f"- R_total = {r_total:.6g} Ω")
        lines.append(f"- I ≈ {i:.6g} A ({i_ma:.6g} mA)")

    # Per-resistor voltage drops (best-effort, using first 2 pins).
    for idx, r in enumerate(rs, start=1):
        comp = idx  # R1 comp_index=1
        if comp + 1 >= len(voltage_ord):
            break
        pin0 = voltage_ord[comp]
        pin1 = voltage_ord[comp + 1]
        if (pin1 - pin0) < 2:
            continue
        v0 = float(voltage[pin0 + 0])
        v1 = float(voltage[pin0 + 1])
        vdrop = abs(v0 - v1)
        if lang_zh:
            lines.append(f"- R{idx}={float(r):.6g}Ω, V_drop≈{vdrop:.6g}V")
        else:
            lines.append(f"- R{idx}={float(r):.6g}Ω, V_drop≈{vdrop:.6g}V")

    return "\n".join(lines)


def simulate_series_vdc_two_resistors(
    *,
    text: str,
    phy_engine_cfg: Any,
    config_base_dir: str,
) -> str:
    spec = _parse_series_vdc_2r_spec(text)
    if spec is None:
        return (
            "To run a DC simulation, include values for V, R1, and R2.\n"
            "Example: 'simulate V=5 R1=100ohm R2=200ohm'"
        )

    cmake_source_dir = _resolve_existing_path(
        str(getattr(phy_engine_cfg, "cmake_source_dir", "")),
        base_dir=config_base_dir,
    )
    cmake_build_dir = os.path.abspath(
        os.path.join(config_base_dir, str(getattr(phy_engine_cfg, "cmake_build_dir", "")))
    )
    lib_cfg_path = _resolve_existing_path(
        str(getattr(phy_engine_cfg, "phyengine_lib_path", "")),
        base_dir=config_base_dir,
    )
    lib_path = ensure_phyengine_lib(
        phyengine_lib_path=lib_cfg_path,
        auto_build=bool(getattr(phy_engine_cfg, "auto_build", False)),
        cmake_source_dir=cmake_source_dir,
        cmake_build_dir=cmake_build_dir,
        cmake_build_type=str(getattr(phy_engine_cfg, "cmake_build_type", "Release")),
        build_timeout_sec=int(getattr(phy_engine_cfg, "build_timeout_sec", 900)),
    )

    pe = PhyEngineLib(lib_path)
    res = pe.simulate_series_vdc_two_resistors(spec=spec)
    i_ma = res.current_a * 1000.0
    vdrop1 = res.v_plus - res.v_mid
    vdrop2 = res.v_mid - res.v_ground
    return (
        "DC simulation (VDC + 2 resistors in series):\n"
        f"- R_total = {res.r_total_ohm:.6g} Ω\n"
        f"- I = {res.current_a:.6g} A ({i_ma:.6g} mA)\n"
        f"- V_drop(R1) ≈ {vdrop1:.6g} V, V_drop(R2) ≈ {vdrop2:.6g} V"
    )


_TIME_RE = re.compile(
    r"(?i)(?:time|t)\\s*=\\s*(?P<value>[0-9]+(?:\\.[0-9]+)?)\\s*(?P<unit>ms|us|µs|ns|s|sec|secs|second|seconds)?"
)


def _parse_time_seconds(text: str) -> float | None:
    text = (text or "").strip()
    if not text:
        return None
    m = _TIME_RE.search(text)
    if not m:
        return None
    v = float(m.group("value"))
    unit = (m.group("unit") or "s").lower()
    if unit in ("s", "sec", "secs", "second", "seconds"):
        return v
    if unit == "ms":
        return v / 1000.0
    if unit in ("us", "µs"):
        return v / 1_000_000.0
    if unit == "ns":
        return v / 1_000_000_000.0
    return v


def _looks_like_zh(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in (text or ""))


_DIGITAL_TICKS_RE = re.compile(
    r"(?i)(?:ticks?|clks?|clock)\s*=\s*(?P<n>[0-9]{1,7})|(?:时钟|脉冲)\s*(?P<n2>[0-9]{1,7})\s*(?:次|下|拍)?"
)


def _parse_digital_ticks(text: str) -> int | None:
    m = _DIGITAL_TICKS_RE.search(text or "")
    if not m:
        return None
    n = m.group("n") or m.group("n2")
    if not n:
        return None
    try:
        v = int(n)
    except Exception:
        return None
    if v < 0:
        return None
    return min(1_000_000, v)


def _status_save_has_logic(status_save: dict[str, Any]) -> bool:
    els = status_save.get("Elements")
    if not isinstance(els, list):
        return False
    for e in els:
        if not isinstance(e, dict):
            continue
        mid = e.get("ModelID")
        if not isinstance(mid, str):
            continue
        s = mid.casefold()
        if "logic" in s or "gate" in s or "flipflop" in s or "adder" in s or "subtractor" in s or "counter" in s:
            return True
    return False


_LOGIC_ASSIGN_RE = re.compile(r"(?i)\b(?P<name>[A-Za-z][A-Za-z0-9_]*)\s*=\s*(?P<val>[01])\b")


def _parse_logic_input_assignments(text: str) -> dict[str, int]:
    assigns: dict[str, int] = {}
    for m in _LOGIC_ASSIGN_RE.finditer(text or ""):
        name = (m.group("name") or "").strip()
        if not name:
            continue
        v = 1 if (m.group("val") or "0") == "1" else 0
        assigns[name.casefold()] = v
    return assigns


def _apply_logic_assignments(status_save: dict[str, Any], assigns: dict[str, int]) -> None:
    if not assigns:
        return
    els = status_save.get("Elements")
    if not isinstance(els, list):
        return
    for el in els:
        if not isinstance(el, dict):
            continue
        if el.get("ModelID") != "Logic Input":
            continue
        label = el.get("Label")
        if not isinstance(label, str) or not label.strip():
            continue
        key = label.strip().casefold()
        if key not in assigns:
            continue
        props = el.get("Properties")
        if not isinstance(props, dict):
            props = {}
            el["Properties"] = props
        props["开关"] = float(assigns[key])


def _load_status_save_from_sav_path(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        root = json.load(f)
    if not isinstance(root, dict):
        raise RuntimeError(".sav root must be an object")
    exp = root.get("Experiment")
    if not isinstance(exp, dict):
        raise RuntimeError(".sav missing Experiment object")
    ss = exp.get("StatusSave")
    if isinstance(ss, str):
        try:
            obj = json.loads(ss)
        except json.JSONDecodeError as e:
            raise RuntimeError(f".sav Experiment.StatusSave is not valid JSON: {e}") from e
        if not isinstance(obj, dict):
            raise RuntimeError(".sav StatusSave JSON is not an object")
        return obj
    if isinstance(ss, dict):
        return ss
    raise RuntimeError(".sav missing Experiment.StatusSave")


def simulate_ai_verilog_with_phyengine(
    *,
    ollama: OllamaClient,
    text: str,
    context_json: dict[str, Any] | None,
    phy_engine_cfg: Any,
    config_base_dir: str,
    cache_dir: str,
    max_attempts: int = 2,
    max_elements: int = 1000,
) -> str:
    """LLM/user Verilog -> verilog2plsav -> StatusSave -> Phy-Engine adapter simulation."""
    lang_zh = _looks_like_zh(text)
    if max_attempts <= 0:
        max_attempts = 1
    if max_attempts > 5:
        max_attempts = 5

    verilog = extract_fenced_code(text, preferred_lang="verilog")
    if not verilog:
        # Heuristic: user pasted Verilog without fences.
        t = (text or "")
        if "module" in t and "endmodule" in t:
            verilog = t
    if not verilog:
        verilog = llm_generate_verilog(ollama=ollama, spec=text)

    cmake_source_dir = _resolve_existing_path(
        str(getattr(phy_engine_cfg, "cmake_source_dir", "")),
        base_dir=config_base_dir,
    )
    verilog2plsav_cfg_path = _resolve_existing_path(
        str(getattr(phy_engine_cfg, "verilog2plsav_path", "")),
        base_dir=config_base_dir,
    )
    verilog2plsav_bin = ensure_verilog2plsav(
        verilog2plsav_path=verilog2plsav_cfg_path,
        auto_build=bool(getattr(phy_engine_cfg, "auto_build", False)),
        cmake_source_dir=cmake_source_dir,
        cmake_build_dir=os.path.abspath(
            os.path.join(config_base_dir, str(getattr(phy_engine_cfg, "cmake_build_dir", "")))
        ),
        cmake_build_type=str(getattr(phy_engine_cfg, "cmake_build_type", "Release")),
        build_timeout_sec=int(getattr(phy_engine_cfg, "build_timeout_sec", 900)),
    )

    extra_args = getattr(phy_engine_cfg, "verilog2plsav_args", None)
    extra_args_list: list[str] = []
    if isinstance(extra_args, list) and all(isinstance(x, str) for x in extra_args):
        extra_args_list = [x for x in extra_args if x.strip()]

    assigns = _parse_logic_input_assignments(text)

    with tempfile.TemporaryDirectory(prefix="phy_lab_verilog_sim_", dir=cache_dir) as temp_dir:
        in_v = os.path.join(temp_dir, "design.v")
        out_sav = os.path.join(temp_dir, "design.sav")

        last_error: str | None = None
        for attempt in range(1, max_attempts + 1):
            with open(in_v, "w", encoding="utf-8") as f:
                f.write(verilog)
                f.write("\n")
            try:
                verilog_to_plsav(
                    verilog2plsav_bin=verilog2plsav_bin,
                    out_sav_path=out_sav,
                    in_verilog_path=in_v,
                    options=Verilog2PlSavOptions(top="top", extra_args=extra_args_list or None),
                    timeout_sec=int(getattr(phy_engine_cfg, "run_timeout_sec", 300)),
                )
                last_error = None
                break
            except Exception as e:
                last_error = str(e) or e.__class__.__name__
                if attempt >= max_attempts:
                    if lang_zh:
                        return "Verilog→.sav 编译失败：\n" + truncate(last_error, max_chars=1800)
                    return "Verilog→.sav compilation failed:\n" + truncate(last_error, max_chars=1800)
                verilog = llm_fix_verilog(
                    ollama=ollama,
                    spec=text,
                    prior_verilog=verilog,
                    error_text=last_error,
                )

        status_save = _load_status_save_from_sav_path(out_sav)
        _apply_logic_assignments(status_save, assigns)

        return simulate_status_save_with_phyengine(
            text=text,
            status_save=status_save,
            phy_engine_cfg=phy_engine_cfg,
            config_base_dir=config_base_dir,
            max_elements=int(max_elements),
        )


def _pe_sim_timeout_sec(phy_engine_cfg: Any, *, default_sec: float = 5.0) -> float:
    v = getattr(phy_engine_cfg, "sim_timeout_sec", None)
    if v is None:
        v = default_sec
    try:
        v = float(v)
    except Exception:
        v = float(default_sec)
    if v <= 0.0:
        v = float(default_sec)
    # Keep it bounded to avoid runaway settings.
    return min(60.0, max(1.0, v))


def _pe_status_save_adapter_worker(
    *,
    lib_path: str,
    status_save: dict[str, Any],
    analyze_type: int,
    tr_t_step_s: float,
    tr_t_stop_s: float,
    ac_omega_rad_s: float,
    digital_clk_ticks: int,
) -> dict[str, Any]:
    lib = PhyEngineLib(lib_path)
    return lib.simulate_status_save(
        status_save=status_save,
        analyze_type=int(analyze_type),
        tr_t_step_s=float(tr_t_step_s),
        tr_t_stop_s=float(tr_t_stop_s),
        ac_omega_rad_s=float(ac_omega_rad_s),
        digital_clk_ticks=int(digital_clk_ticks),
        indent=0,
    )


def _pe_analyze_and_sample_worker(
    *,
    lib_path: str,
    element_codes: list[int],
    wires: list[int],
    properties: list[float],
    analyze_type: int,
    tr_t_step_s: float,
    tr_t_stop_s: float,
    ac_omega_rad_s: float,
    max_pins_per_comp: int,
    max_branches_per_comp: int,
) -> dict[str, Any]:
    pe = PhyEngineLib(lib_path)
    circuit, vec_pos, chunk_pos, comp_size = pe.create_circuit(
        element_codes=element_codes,
        wires=wires,
        properties=properties,
    )
    try:
        pe.set_analyze_type(circuit=circuit, analyze_type=int(analyze_type))
        if int(analyze_type) in (AnalyzeType.TR, AnalyzeType.TROP):
            pe.set_tr(
                circuit=circuit,
                t_step=float(tr_t_step_s),
                t_stop=float(tr_t_stop_s),
            )
        elif int(analyze_type) in (AnalyzeType.AC, AnalyzeType.ACOP):
            pe.set_ac_omega(
                circuit=circuit,
                omega=float(ac_omega_rad_s),
            )
        pe.analyze(circuit=circuit)
        voltage, voltage_ord, current, current_ord, _digital, _digital_ord = pe.sample(
            circuit=circuit,
            vec_pos=vec_pos,
            chunk_pos=chunk_pos,
            comp_size=comp_size,
            max_pins_per_comp=int(max_pins_per_comp),
            max_branches_per_comp=int(max_branches_per_comp),
        )
        return {
            "comp_size": int(comp_size),
            "voltage": voltage,
            "voltage_ord": voltage_ord,
            "current": current,
            "current_ord": current_ord,
        }
    finally:
        pe.destroy_circuit(circuit=circuit, vec_pos=vec_pos, chunk_pos=chunk_pos)


def simulate_status_save_with_phyengine(
    *,
    text: str,
    status_save: dict[str, Any],
    phy_engine_cfg: Any,
    config_base_dir: str,
    max_elements: int = 300,
) -> str:
    lang_zh = _looks_like_zh(text)
    elements = status_save.get("Elements")
    wires = status_save.get("Wires")
    if not isinstance(elements, list) or not isinstance(wires, list):
        return "StatusSave is missing Elements/Wires."

    if max_elements > 0 and len(elements) > max_elements:
        if lang_zh:
            return (
                f"该实验电路规模过大（elements={len(elements)}，limit={max_elements}），为安全起见我不会本地仿真。\n"
                "请提供一个更小的电路（仅包含电阻/电容/电感/电源/地），或者把关键部分单独做成小实验。"
            )
        return (
            f"This experiment is too large to simulate safely (elements={len(elements)}, limit={max_elements}).\n"
            "Please provide a smaller circuit (R/C/L/sources/ground only) or isolate the critical sub-circuit."
        )

    cmake_source_dir = _resolve_existing_path(
        str(getattr(phy_engine_cfg, "cmake_source_dir", "")),
        base_dir=config_base_dir,
    )
    lib_cfg_path = _resolve_existing_path(
        str(getattr(phy_engine_cfg, "phyengine_lib_path", "")),
        base_dir=config_base_dir,
    )
    lib_path = ensure_phyengine_lib(
        phyengine_lib_path=lib_cfg_path,
        auto_build=bool(getattr(phy_engine_cfg, "auto_build", False)),
        cmake_source_dir=cmake_source_dir,
        cmake_build_dir=os.path.abspath(
            os.path.join(config_base_dir, str(getattr(phy_engine_cfg, "cmake_build_dir", "")))
        ),
        cmake_build_type=str(getattr(phy_engine_cfg, "cmake_build_type", "Release")),
        build_timeout_sec=int(getattr(phy_engine_cfg, "build_timeout_sec", 900)),
    )
    lib = PhyEngineLib(lib_path)
    timeout_sec = _pe_sim_timeout_sec(phy_engine_cfg, default_sec=5.0)

    t_stop = _parse_time_seconds(text)
    wants_tr = t_stop is not None or any(
        x in (text or "").lower() for x in ("transient", "tr", "瞬态", "时域")
    )
    if t_stop is None and wants_tr:
        t_stop = 1e-3
    if t_stop is not None and t_stop <= 0:
        t_stop = 1e-3
    t_step = None
    if t_stop is not None:
        t_step = min(1e-3, max(1e-9, t_stop / 2000.0))

    analyze_type = 4 if wants_tr else 1
    adapter_err = ""
    if lib.can_simulate_status_save():
        ticks = _parse_digital_ticks(text)
        if ticks is None:
            ticks = 1 if _status_save_has_logic(status_save) else 0
        try:
            out_status = run_with_timeout(
                fn=_pe_status_save_adapter_worker,
                kwargs={
                    "lib_path": lib_path,
                    "status_save": status_save,
                    "analyze_type": int(analyze_type),
                    "tr_t_step_s": float(t_step or 1e-6) if wants_tr else 0.0,
                    "tr_t_stop_s": float(t_stop or 1e-6) if wants_tr else 0.0,
                    "ac_omega_rad_s": 0.0,
                    "digital_clk_ticks": int(ticks),
                },
                timeout_sec=float(timeout_sec),
                label="Phy-Engine StatusSave simulation",
            )
        except (PESimError, ProcRemoteError) as e:
            # Fall back to the Python mapper for simple circuits if the adapter fails.
            out_status = None
            adapter_err = str(e)
        except ProcTimeoutError as e:
            if lang_zh:
                return f"Phy-Engine 仿真超时（>{timeout_sec:g}s），已中止以避免卡住后续任务。"
            return f"Phy-Engine simulation timed out (>{timeout_sec:g}s); aborted to avoid blocking later tasks."
        else:
            adapter_err = ""

        if isinstance(out_status, dict):
            wants_caps = ("capacitor" in (text or "").lower()) or ("电容" in (text or ""))
            wants_res = ("resistor" in (text or "").lower()) or ("电阻" in (text or ""))
            wants_ind = ("inductor" in (text or "").lower()) or ("电感" in (text or ""))

            def _pick(model_id: str) -> bool:
                if wants_caps:
                    return "Capacitor" in model_id
                if wants_res:
                    return "Resistor" in model_id
                if wants_ind:
                    return "Inductor" in model_id
                return True

            if lang_zh:
                header = "仿真结果" + ("（TR）" if wants_tr else "（DC）")
                if wants_tr and t_stop is not None:
                    header += f"  t_stop={t_stop:g}s"
            else:
                header = "Simulation result" + (" (TR)" if wants_tr else " (DC)")
                if wants_tr and t_stop is not None:
                    header += f"  t_stop={t_stop:g}s"

            els = out_status.get("Elements")
            if not isinstance(els, list):
                return header + ("\nStatusSave.Elements missing." if not lang_zh else "\nStatusSave.Elements 缺失。")

            rows: list[dict[str, Any]] = []
            for el in els:
                if not isinstance(el, dict):
                    continue
                model_id = el.get("ModelID") if isinstance(el.get("ModelID"), str) else ""
                if model_id and not _pick(model_id):
                    continue
                st = el.get("Statistics") if isinstance(el.get("Statistics"), dict) else {}
                props = el.get("Properties") if isinstance(el.get("Properties"), dict) else {}
                ident = el.get("Identifier") if isinstance(el.get("Identifier"), str) else ""
                label = el.get("Label") if isinstance(el.get("Label"), str) else ""
                rows.append(
                    {
                        "identifier": ident.strip(),
                        "label": label.strip(),
                        "model_id": model_id.strip(),
                        "v": st.get("电压"),
                        "i": st.get("电流"),
                        "p": st.get("功率"),
                        "state": props.get("状态"),
                    }
                )

            if not rows:
                rows = []
                for el in els:
                    if not isinstance(el, dict):
                        continue
                    model_id = el.get("ModelID") if isinstance(el.get("ModelID"), str) else ""
                    st = el.get("Statistics") if isinstance(el.get("Statistics"), dict) else {}
                    props = el.get("Properties") if isinstance(el.get("Properties"), dict) else {}
                    ident = el.get("Identifier") if isinstance(el.get("Identifier"), str) else ""
                    label = el.get("Label") if isinstance(el.get("Label"), str) else ""
                    rows.append(
                        {
                            "identifier": ident.strip(),
                            "label": label.strip(),
                            "model_id": model_id.strip(),
                            "v": st.get("电压"),
                            "i": st.get("电流"),
                            "p": st.get("功率"),
                            "state": props.get("状态"),
                        }
                    )

            lines: list[str] = [header]
            limit = 20
            shown = 0
            for r in rows:
                if shown >= limit:
                    break
                name_bits = [b for b in [r.get("identifier"), r.get("label"), r.get("model_id")] if isinstance(b, str) and b]
                name = " ".join(name_bits) if name_bits else "(element)"
                v = r.get("v")
                i = r.get("i")
                p = r.get("p")
                state = r.get("state")
                parts: list[str] = []
                if isinstance(v, (int, float)):
                    parts.append(f"V≈{float(v):.6g}V")
                if isinstance(i, (int, float)):
                    parts.append(f"I≈{float(i):.6g}A")
                if isinstance(p, (int, float)):
                    parts.append(f"P≈{float(p):.6g}W")
                if isinstance(state, (int, float)):
                    parts.append(f"state={int(state)}")
                if not parts:
                    continue
                lines.append(f"- {name}: " + " ".join(parts))
                shown += 1

            if len(rows) > shown:
                remaining = len(rows) - shown
                if lang_zh:
                    lines.append(f"(还有 {remaining} 个元件未展示；你可以在问题里点名例如 '电容'/'电阻' 来筛选)")
                else:
                    lines.append(f"({remaining} more elements not shown; mention 'capacitor'/'resistor' to filter)")

            if adapter_err:
                # Keep the adapter error for debugging only if we showed something useful.
                pass
            return "\n".join(lines)

    try:
        pe_input = build_pe_circuit_input_from_status_save(status_save, strict=True)
    except PlSavSimError as e:
        if adapter_err:
            if lang_zh:
                return f"StatusSave→Phy-Engine 适配器失败：{adapter_err}\n回退到简化映射也失败：{e}"
            return f"StatusSave→Phy-Engine adapter failed: {adapter_err}\nFallback mapper also failed: {e}"
        return str(e)

    try:
        sim_out = run_with_timeout(
            fn=_pe_analyze_and_sample_worker,
            kwargs={
                "lib_path": lib_path,
                "element_codes": pe_input.element_codes,
                "wires": pe_input.wires,
                "properties": pe_input.properties,
                "analyze_type": int(analyze_type),
                "tr_t_step_s": float(t_step or 1e-6) if wants_tr else 0.0,
                "tr_t_stop_s": float(t_stop or 1e-6) if wants_tr else 0.0,
                "ac_omega_rad_s": 0.0,
                "max_pins_per_comp": 8,
                "max_branches_per_comp": 4,
            },
            timeout_sec=float(timeout_sec),
            label="Phy-Engine simulation",
        )
    except ProcTimeoutError:
        if lang_zh:
            return f"Phy-Engine 仿真超时（>{timeout_sec:g}s），已中止以避免卡住后续任务。"
        return f"Phy-Engine simulation timed out (>{timeout_sec:g}s); aborted to avoid blocking later tasks."
    except ProcRemoteError as e:
        if lang_zh:
            return f"Phy-Engine 仿真失败：{e}"
        return f"Phy-Engine simulation failed: {e}"

    voltage = list(sim_out.get("voltage") or [])
    voltage_ord = list(sim_out.get("voltage_ord") or [])
    current = list(sim_out.get("current") or [])
    current_ord = list(sim_out.get("current_ord") or [])

    wants_caps = ("capacitor" in (text or "").lower()) or ("电容" in (text or ""))
    wants_res = ("resistor" in (text or "").lower()) or ("电阻" in (text or ""))
    wants_ind = ("inductor" in (text or "").lower()) or ("电感" in (text or ""))

    def _pick(model_id: str) -> bool:
        if wants_caps:
            return "Capacitor" in model_id
        if wants_res:
            return "Resistor" in model_id
        if wants_ind:
            return "Inductor" in model_id
        return True

    metas = [m for m in pe_input.non_ground_meta if _pick(m.model_id)]
    if not metas:
        metas = list(pe_input.non_ground_meta)

    if lang_zh:
        header = "仿真结果" + ("（TR）" if wants_tr else "（DC）")
        if wants_tr and t_stop is not None:
            header += f"  t_stop={t_stop:g}s"
    else:
        header = "Simulation result" + (" (TR)" if wants_tr else " (DC)")
        if wants_tr and t_stop is not None:
            header += f"  t_stop={t_stop:g}s"

    lines: list[str] = [header]
    limit = 20
    for meta in metas[:limit]:
        comp_id = int(getattr(meta, "comp_index", 0))
        if comp_id < 0 or comp_id + 1 >= len(voltage_ord) or comp_id + 1 >= len(current_ord):
            continue
        pin0 = voltage_ord[comp_id]
        pin1 = voltage_ord[comp_id + 1]
        cur0 = current_ord[comp_id]
        cur1 = current_ord[comp_id + 1]
        vs = voltage[pin0:pin1]
        cs = current[cur0:cur1]
        label = f"{meta.label} " if meta.label else ""
        name = f"{label}{meta.model_id}"
        v_str = ", ".join(f"{v:.6g}V" for v in vs[:4])
        c_str = ", ".join(f"{c:.6g}A" for c in cs[:2])
        lines.append(f"- {name}: V=[{v_str}] I=[{c_str}]")

    if len(metas) > limit:
        remaining = len(metas) - limit
        if lang_zh:
            lines.append(f"(还有 {remaining} 个元件未展示；你可以在问题里点名例如 'C1' 或 'Resistor' 来筛选)")
        else:
            lines.append(
                f"({remaining} more components not shown; mention a label like 'C1' or a type like 'Resistor' to filter)"
            )

    return "\n".join(lines)


_GOOGLE_RESULT_RE = re.compile(
    r'href="/url\\?q=(?P<url>[^"&]+)[^"]*"[^>]*>(?P<title>[^<]{3,200})<',
    re.IGNORECASE,
)

_GOOGLE_H3_RE = re.compile(
    r'<a[^>]+href="/url\\?q=(?P<url>[^"&]+)[^"]*"[^>]*>\\s*<h3[^>]*>(?P<title>.*?)</h3>',
    re.IGNORECASE | re.DOTALL,
)


def _strip_tags(s: str) -> str:
    s = re.sub(r"<[^>]+>", " ", s or "")
    s = html.unescape(s)
    s = re.sub(r"\\s+", " ", s).strip()
    return s


def _looks_like_google_block(html_text: str) -> bool:
    t = (html_text or "").casefold()
    return any(
        x in t
        for x in (
            "our systems have detected unusual traffic",
            "/sorry/",
            "recaptcha",
            "captcha",
            "consent.google.com",
            "unusual traffic",
        )
    )


def _cache_get(path: str, *, ttl_sec: int) -> str | None:
    try:
        st = os.stat(path)
    except OSError:
        return None
    if ttl_sec > 0 and (time.time() - st.st_mtime) > ttl_sec:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except OSError:
        return None


def _cache_put(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(content)
    os.replace(tmp, path)


def web_search_google(
    *,
    query: str,
    cache_dir: str,
    proxy: str = "",
    timeout_sec: int = 20,
    ttl_sec: int = 3600,
    max_results: int = 5,
    user_agent: str = "",
) -> str:
    query = (query or "").strip()
    if not query:
        return "Provide a query string."
    if max_results <= 0:
        max_results = 5
    if max_results > 10:
        max_results = 10

    url = "https://www.google.com/search?" + urllib.parse.urlencode(
        {
            "q": query,
            "num": str(max_results),
            "hl": "en",
        }
    )

    cache_root = os.path.join(cache_dir, "web_cache", "google")
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    cache_path = os.path.join(cache_root, f"{digest}.html")
    cached = _cache_get(cache_path, ttl_sec=int(ttl_sec))
    if cached is None or _looks_like_google_block(cached):
        cached = _http_get_text(
            url=url,
            proxy=proxy,
            timeout_sec=float(timeout_sec),
            user_agent=user_agent,
        )
        if not _looks_like_google_block(cached):
            _cache_put(cache_path, cached)

    results: list[tuple[str, str]] = []
    for m in _GOOGLE_H3_RE.finditer(cached):
        raw_url = m.group("url") or ""
        title = _strip_tags(m.group("title") or "")
        try:
            target_url = urllib.parse.unquote(raw_url)
        except Exception:
            target_url = raw_url
        if not target_url.startswith("http"):
            continue
        if not title:
            continue
        results.append((title, target_url))
        if len(results) >= max_results:
            break

    if not results:
        for m in _GOOGLE_RESULT_RE.finditer(cached):
            raw_url = m.group("url") or ""
            title = _strip_tags(m.group("title") or "")
            try:
                target_url = urllib.parse.unquote(raw_url)
            except Exception:
                target_url = raw_url
            if not target_url.startswith("http"):
                continue
            if not title:
                continue
            results.append((title, target_url))
            if len(results) >= max_results:
                break

    if not results:
        if _looks_like_google_block(cached):
            return (
                "Google blocked this automated request (captcha/consent/unusual-traffic).\n"
                "Try: enable proxy, change IP, reduce frequency, or use the DuckDuckGo fallback."
            )
        return (
            "No results parsed. Google may have returned a blocked/captcha page or changed markup.\n"
            f"Query: {query}"
        )

    lines = ["Google results:"]
    for i, (t, u) in enumerate(results, start=1):
        lines.append(f"{i}. {truncate(t, max_chars=120)}")
        lines.append(f"   {u}")
    return "\n".join(lines)


def web_search_duckduckgo(
    *,
    query: str,
    cache_dir: str,
    proxy: str = "",
    timeout_sec: int = 20,
    ttl_sec: int = 3600,
    max_results: int = 5,
    user_agent: str = "",
) -> str:
    lib_error_note: str | None = None

    def _try_ddgsearch() -> str | None:
        """DuckDuckGo via the `duckduckgo-search` library (preferred).

        Returns:
            - str: formatted results
            - None: if dependency is missing (so caller can fall back)
        """
        nonlocal lib_error_note
        try:
            from duckduckgo_search import DDGS  # type: ignore
        except Exception:
            return None

        import inspect

        q = (query or "").strip()
        if not q:
            return "Provide a query string."

        if max_results <= 0:
            n = 5
        else:
            n = int(max_results)
        if n > 10:
            n = 10

        cache_root = os.path.join(cache_dir, "web_cache", "duckduckgo_search")
        digest = hashlib.sha256(
            json.dumps({"q": q, "n": n}, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        cache_path = os.path.join(cache_root, f"{digest}.json")
        cached = _cache_get(cache_path, ttl_sec=int(ttl_sec))
        if cached is not None:
            try:
                data = json.loads(cached)
                if isinstance(data, list):
                    items = [x for x in data if isinstance(x, dict)]
                else:
                    items = []
            except Exception:
                items = []
        else:
            kwargs: dict[str, Any] = {}
            try:
                sig = inspect.signature(DDGS)
                params = sig.parameters
            except Exception:
                params = {}

            p = (proxy or "").strip()
            if p:
                if "://" not in p:
                    p = "http://" + p
                if "proxy" in params:
                    kwargs["proxy"] = p
                elif "proxies" in params:
                    kwargs["proxies"] = {"http": p, "https": p}

            ua = (user_agent or "").strip() or _default_user_agent()
            if "headers" in params:
                kwargs["headers"] = {"User-Agent": ua}

            if "timeout" in params:
                kwargs["timeout"] = float(timeout_sec)

        try:
            ddgs = DDGS(**kwargs) if kwargs else DDGS()
        except RecursionError as e:
            # Some environments / library versions can trigger recursive proxy/session wiring.
            lib_error_note = f"duckduckgo-search init failed: {e}"
            return None
        try:
            results = ddgs.text(q, max_results=n)  # type: ignore[call-arg]
        except TypeError:
            results = ddgs.text(q, n)  # type: ignore[misc]
        except RecursionError as e:
            lib_error_note = f"duckduckgo-search failed: {e}"
            return None

        try:
            raw_items = list(results)
        except TypeError:
            raw_items = results if isinstance(results, list) else []
        except RecursionError as e:
            lib_error_note = f"duckduckgo-search failed: {e}"
            return None

        items = [x for x in raw_items if isinstance(x, dict)]
        _cache_put(cache_path, json.dumps(items, ensure_ascii=False))

        pairs: list[tuple[str, str]] = []
        seen: set[str] = set()
        for it in items:
            title = str(it.get("title") or it.get("heading") or it.get("name") or "").strip()
            url = str(it.get("href") or it.get("url") or it.get("link") or "").strip()
            if not title or not url or not url.startswith("http"):
                continue
            if url in seen:
                continue
            seen.add(url)
            pairs.append((title, url))
            if len(pairs) >= n:
                break

        if not pairs:
            return f"No results parsed from DuckDuckGo (duckduckgo-search).\nQuery: {q}"

        lines = ["DuckDuckGo results:"]
        for i, (t, u) in enumerate(pairs, start=1):
            lines.append(f"{i}. {truncate(t, max_chars=120)}")
            lines.append(f"   {u}")
        return "\n".join(lines)

    query = (query or "").strip()
    if not query:
        return "Provide a query string."
    if max_results <= 0:
        max_results = 5
    if max_results > 10:
        max_results = 10

    ddgsearch_res = _try_ddgsearch()
    if ddgsearch_res is not None:
        return ddgsearch_res

    # Prefer the lightweight HTML endpoint (often less blocked than the main site).
    url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})

    cache_root = os.path.join(cache_dir, "web_cache", "duckduckgo")
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    cache_path = os.path.join(cache_root, f"{digest}.html")
    cached = _cache_get(cache_path, ttl_sec=int(ttl_sec))
    if cached is None:
        cached = _http_get_text(
            url=url,
            proxy=proxy,
            timeout_sec=float(timeout_sec),
            user_agent=user_agent,
        )
        _cache_put(cache_path, cached)

    a_re = re.compile(
        r'<a[^>]+class="result__a"[^>]+href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>',
        re.IGNORECASE | re.DOTALL,
    )
    results: list[tuple[str, str]] = []
    for m in a_re.finditer(cached):
        u = html.unescape((m.group("url") or "").strip())
        t = _strip_tags(m.group("title") or "")
        if not u.startswith("http"):
            continue
        if not t:
            continue
        results.append((t, u))
        if len(results) >= max_results:
            break

    if not results:
        extra = ""
        if lib_error_note:
            extra = f"\nNote: {lib_error_note}"
        return f"No results parsed from DuckDuckGo.\nQuery: {query}{extra}"

    lines = ["DuckDuckGo results:"]
    if lib_error_note:
        lines.append(f"(Note: {lib_error_note}; fell back to HTML endpoint)")
    for i, (t, u) in enumerate(results, start=1):
        lines.append(f"{i}. {truncate(t, max_chars=120)}")
        lines.append(f"   {u}")
    return "\n".join(lines)


_BING_H2_RE = re.compile(
    r'<li[^>]*class="[^"]*b_algo[^"]*"[^>]*>.*?<h2[^>]*>\\s*<a[^>]+href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)


def _looks_like_bing_block(html_text: str) -> bool:
    t = (html_text or "").casefold()
    return any(
        x in t
        for x in (
            "unusual traffic",
            "verify you are a human",
            "captcha",
            "our systems have detected unusual traffic",
        )
    )


def web_search_bing(
    *,
    query: str,
    cache_dir: str,
    proxy: str = "",
    timeout_sec: int = 20,
    ttl_sec: int = 3600,
    max_results: int = 5,
    user_agent: str = "",
) -> str:
    query = (query or "").strip()
    if not query:
        return "Provide a query string."
    if max_results <= 0:
        max_results = 5
    if max_results > 10:
        max_results = 10

    url = "https://www.bing.com/search?" + urllib.parse.urlencode(
        {
            "q": query,
            "count": str(max_results),
            "setlang": "en-us",
        }
    )

    cache_root = os.path.join(cache_dir, "web_cache", "bing")
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    cache_path = os.path.join(cache_root, f"{digest}.html")
    cached = _cache_get(cache_path, ttl_sec=int(ttl_sec))
    if cached is None or _looks_like_bing_block(cached):
        cached = _http_get_text(
            url=url,
            proxy=proxy,
            timeout_sec=float(timeout_sec),
            user_agent=user_agent,
        )
        if not _looks_like_bing_block(cached):
            _cache_put(cache_path, cached)

    results: list[tuple[str, str]] = []
    for m in _BING_H2_RE.finditer(cached):
        u = html.unescape((m.group("url") or "").strip())
        t = _strip_tags(m.group("title") or "")
        if not u.startswith("http"):
            continue
        if not t:
            continue
        results.append((t, u))
        if len(results) >= max_results:
            break

    if not results:
        if _looks_like_bing_block(cached):
            return (
                "Bing blocked this automated request (captcha/verification).\n"
                "Try: enable proxy, change IP, reduce frequency, or use the DuckDuckGo fallback."
            )
        return (
            "No results parsed. Bing may have returned a blocked/captcha page or changed markup.\n"
            f"Query: {query}"
        )

    lines = ["Bing results:"]
    for i, (t, u) in enumerate(results, start=1):
        lines.append(f"{i}. {truncate(t, max_chars=120)}")
        lines.append(f"   {u}")
    return "\n".join(lines)


def web_search_searxng(
    *,
    query: str,
    cache_dir: str,
    base_url: str = "",
    proxy: str = "",
    timeout_sec: int = 20,
    ttl_sec: int = 3600,
    max_results: int = 5,
    user_agent: str = "",
) -> str:
    """Search via a SearXNG instance (recommended for robustness).

    Requires a running SearXNG server reachable from this machine.
    """
    query = (query or "").strip()
    if not query:
        return "Provide a query string."
    if max_results <= 0:
        max_results = 5
    if max_results > 10:
        max_results = 10

    b = (base_url or "").strip().rstrip("/")
    if not b:
        b = "http://127.0.0.1:8080"

    url = b + "/search?" + urllib.parse.urlencode(
        {
            "q": query,
            "format": "json",
            "language": "auto",
        }
    )

    cache_root = os.path.join(cache_dir, "web_cache", "searxng")
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    cache_path = os.path.join(cache_root, f"{digest}.json")
    cached = _cache_get(cache_path, ttl_sec=int(ttl_sec))
    if cached is None:
        cached = _http_get_text(
            url=url,
            proxy=proxy,
            timeout_sec=float(timeout_sec),
            user_agent=user_agent,
        )
        _cache_put(cache_path, cached)

    try:
        data = json.loads(cached)
    except Exception as e:
        return f"SearXNG returned invalid JSON: {e}"

    items = data.get("results")
    if not isinstance(items, list):
        return "SearXNG returned an unexpected response (missing 'results')."

    results: list[tuple[str, str]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        title = it.get("title")
        href = it.get("url")
        if not isinstance(title, str) or not title.strip():
            continue
        if not isinstance(href, str) or not href.strip():
            continue
        results.append((title.strip(), href.strip()))
        if len(results) >= max_results:
            break

    if not results:
        return (
            "No results parsed from SearXNG.\n"
            f"BaseURL: {b}\n"
            f"Query: {query}"
        )

    lines = ["SearXNG results:"]
    for i, (t, u) in enumerate(results, start=1):
        lines.append(f"{i}. {truncate(t, max_chars=120)}")
        lines.append(f"   {u}")
    return "\n".join(lines)


def web_search(
    *,
    query: str,
    cache_dir: str,
    provider: str = "google",
    proxy: str = "",
    timeout_sec: int = 20,
    ttl_sec: int = 3600,
    max_results: int = 5,
    fallback_to_ddg: bool = True,
    user_agent: str = "",
    searxng_base_url: str = "",
) -> str:
    provider = (provider or "google").strip().lower()
    if provider in ("ddg", "duckduckgo-search", "duckduckgo_search", "ddg-search", "ddg_search", "ddgsearch"):
        provider = "duckduckgo"

    if provider == "baidu":
        res = web_search_baidu(
            query=query,
            cache_dir=cache_dir,
            proxy=proxy,
            timeout_sec=timeout_sec,
            ttl_sec=ttl_sec,
            max_results=max_results,
            user_agent=user_agent,
        )
        if fallback_to_ddg and (
            "Baidu blocked" in res or "No results parsed." in res or "captcha" in res.casefold()
        ):
            try:
                return web_search_duckduckgo(
                    query=query,
                    cache_dir=cache_dir,
                    proxy=proxy,
                    timeout_sec=timeout_sec,
                    ttl_sec=ttl_sec,
                    max_results=max_results,
                    user_agent=user_agent,
                )
            except Exception:
                return res
        return res

    if provider in ("searx", "searxng"):
        return web_search_searxng(
            query=query,
            cache_dir=cache_dir,
            base_url=searxng_base_url,
            proxy=proxy,
            timeout_sec=timeout_sec,
            ttl_sec=ttl_sec,
            max_results=max_results,
            user_agent=user_agent,
        )

    if provider in ("bing", "bing_html"):
        res = web_search_bing(
            query=query,
            cache_dir=cache_dir,
            proxy=proxy,
            timeout_sec=timeout_sec,
            ttl_sec=ttl_sec,
            max_results=max_results,
            user_agent=user_agent,
        )
        if fallback_to_ddg and ("No results parsed." in res or "blocked" in res.casefold()):
            try:
                return web_search_duckduckgo(
                    query=query,
                    cache_dir=cache_dir,
                    proxy=proxy,
                    timeout_sec=timeout_sec,
                    ttl_sec=ttl_sec,
                    max_results=max_results,
                    user_agent=user_agent,
                )
            except Exception:
                return res
        return res

    if provider == "duckduckgo":
        return web_search_duckduckgo(
            query=query,
            cache_dir=cache_dir,
            proxy=proxy,
            timeout_sec=timeout_sec,
            ttl_sec=ttl_sec,
            max_results=max_results,
            user_agent=user_agent,
        )

    res = web_search_google(
        query=query,
        cache_dir=cache_dir,
        proxy=proxy,
        timeout_sec=timeout_sec,
        ttl_sec=ttl_sec,
        max_results=max_results,
        user_agent=user_agent,
    )
    if fallback_to_ddg and (
        "Google blocked this automated request" in res or "No results parsed." in res
    ):
        try:
            return web_search_duckduckgo(
                query=query,
                cache_dir=cache_dir,
                proxy=proxy,
                timeout_sec=timeout_sec,
                ttl_sec=ttl_sec,
                max_results=max_results,
                user_agent=user_agent,
            )
        except Exception:
            return res
    return res


def _looks_like_baidu_block(html_text: str) -> bool:
    t = (html_text or "").casefold()
    return any(
        x in t
        for x in (
            "百度安全验证",
            "请输入验证码",
            "验证码",
            "verify.baidu.com",
            "wappass.baidu.com",
            "captcha",
        )
    )


_BAIDU_H3_RE = re.compile(
    r'<h3[^>]*class="[^"]*(?:t|c-title)[^"]*"[^>]*>\\s*<a[^>]+href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)


def web_search_baidu(
    *,
    query: str,
    cache_dir: str,
    proxy: str = "",
    timeout_sec: int = 20,
    ttl_sec: int = 3600,
    max_results: int = 5,
    user_agent: str = "",
) -> str:
    query = (query or "").strip()
    if not query:
        return "Provide a query string."
    if max_results <= 0:
        max_results = 5
    if max_results > 10:
        max_results = 10

    url = "https://www.baidu.com/s?" + urllib.parse.urlencode(
        {
            "wd": query,
            "rn": str(max_results),
        }
    )

    cache_root = os.path.join(cache_dir, "web_cache", "baidu")
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    cache_path = os.path.join(cache_root, f"{digest}.html")
    cached = _cache_get(cache_path, ttl_sec=int(ttl_sec))
    if cached is None or _looks_like_baidu_block(cached):
        cached = _http_get_text(
            url=url,
            proxy=proxy,
            timeout_sec=float(timeout_sec),
            user_agent=user_agent,
        )
        if not _looks_like_baidu_block(cached):
            _cache_put(cache_path, cached)

    results: list[tuple[str, str]] = []
    for m in _BAIDU_H3_RE.finditer(cached):
        href = html.unescape((m.group("url") or "").strip())
        title = _strip_tags(m.group("title") or "")
        if not href or not title:
            continue
        if href.startswith("//"):
            href = "https:" + href
        results.append((title, href))
        if len(results) >= max_results:
            break

    if not results:
        if _looks_like_baidu_block(cached):
            return (
                "Baidu blocked this automated request (captcha/verification).\n"
                "Try: enable proxy, change IP, reduce frequency, or use the DuckDuckGo fallback."
            )
        return (
            "No results parsed. Baidu may have returned a blocked/captcha page or changed markup.\n"
            f"Query: {query}"
        )

    lines = ["Baidu results:"]
    for i, (t, u) in enumerate(results, start=1):
        lines.append(f"{i}. {truncate(t, max_chars=120)}")
        lines.append(f"   {u}")
    return "\n".join(lines)


@dataclass(frozen=True)
class CircuitBuildResult:
    published: bool
    summary_id: str | None = None
    artifact_dir: str | None = None
    artifact_verilog_path: str | None = None
    artifact_sav_path: str | None = None
    publish_block_reason: str | None = None
    plsav_elements: int | None = None


def _write_artifacts(
    *,
    cache_dir: str,
    verilog_text: str,
    sav_path: str,
) -> tuple[str, str, str]:
    artifacts_root = os.path.join(cache_dir, "artifacts")
    os.makedirs(artifacts_root, exist_ok=True)
    artifact_id = uuid.uuid4().hex
    artifact_dir = os.path.join(artifacts_root, artifact_id)
    os.makedirs(artifact_dir, exist_ok=True)

    v_path = os.path.join(artifact_dir, "design.v")
    out_sav = os.path.join(artifact_dir, "design.sav")
    with open(v_path, "w", encoding="utf-8") as f:
        f.write(verilog_text)
        f.write("\n")
    shutil.copy2(sav_path, out_sav)
    return artifact_dir, v_path, out_sav


def _write_compile_failure_artifacts(
    *,
    cache_dir: str,
    spec: str,
    verilog_text: str,
    error_text: str,
) -> tuple[str, str]:
    artifacts_root = os.path.join(cache_dir, "artifacts")
    os.makedirs(artifacts_root, exist_ok=True)
    artifact_id = uuid.uuid4().hex
    artifact_dir = os.path.join(artifacts_root, artifact_id)
    os.makedirs(artifact_dir, exist_ok=True)

    v_path = os.path.join(artifact_dir, "design.v")
    err_path = os.path.join(artifact_dir, "compile_error.txt")
    spec_path = os.path.join(artifact_dir, "spec.txt")
    with open(v_path, "w", encoding="utf-8") as f:
        f.write(verilog_text)
        f.write("\n")
    with open(err_path, "w", encoding="utf-8") as f:
        f.write(error_text or "")
        f.write("\n")
    with open(spec_path, "w", encoding="utf-8") as f:
        f.write(spec or "")
        f.write("\n")

    return artifact_id, artifact_dir


def build_and_maybe_publish_circuit(
    *,
    ollama: OllamaClient,
    user: Any,
    spec: str,
    cache_dir: str,
    phy_engine_cfg: Any,
    config_base_dir: str,
    keep_temp: bool,
    enable_publish: bool,
    dry_run: bool,
    max_attempts: int = 3,
    publish_max_elements: int = 5000,
    title: str,
    introduction: str,
    publish_category_value: str = "Discussion",
    publish_tags: list[str] | None = None,
) -> CircuitBuildResult:
    os.makedirs(cache_dir, exist_ok=True)

    if max_attempts <= 0:
        max_attempts = 1

    verilog = llm_generate_verilog(ollama=ollama, spec=spec)
    cmake_source_dir = _resolve_existing_path(
        str(getattr(phy_engine_cfg, "cmake_source_dir", "")),
        base_dir=config_base_dir,
    )
    verilog2plsav_cfg_path = _resolve_existing_path(
        str(getattr(phy_engine_cfg, "verilog2plsav_path", "")),
        base_dir=config_base_dir,
    )
    verilog2plsav_bin = ensure_verilog2plsav(
        verilog2plsav_path=verilog2plsav_cfg_path,
        auto_build=bool(getattr(phy_engine_cfg, "auto_build", False)),
        cmake_source_dir=cmake_source_dir,
        cmake_build_dir=os.path.abspath(
            os.path.join(config_base_dir, str(getattr(phy_engine_cfg, "cmake_build_dir", "")))
        ),
        cmake_build_type=str(getattr(phy_engine_cfg, "cmake_build_type", "Release")),
        build_timeout_sec=int(getattr(phy_engine_cfg, "build_timeout_sec", 900)),
    )

    extra_args = getattr(phy_engine_cfg, "verilog2plsav_args", None)
    extra_args_list: list[str] = []
    if isinstance(extra_args, list) and all(isinstance(x, str) for x in extra_args):
        extra_args_list = [x for x in extra_args if x.strip()]

    last_error: str | None = None
    out_sav = ""

    with tempfile.TemporaryDirectory(prefix="phy_lab_", dir=cache_dir) as temp_dir:
        in_v = os.path.join(temp_dir, "design.v")
        out_sav = os.path.join(temp_dir, "design.sav")

        for attempt in range(1, max_attempts + 1):
            with open(in_v, "w", encoding="utf-8") as f:
                f.write(verilog)
                f.write("\n")
            try:
                verilog_to_plsav(
                    verilog2plsav_bin=verilog2plsav_bin,
                    out_sav_path=out_sav,
                    in_verilog_path=in_v,
                    options=Verilog2PlSavOptions(top="top", extra_args=extra_args_list or None),
                    timeout_sec=int(getattr(phy_engine_cfg, "run_timeout_sec", 300)),
                )
                last_error = None
                break
            except Exception as e:
                last_error = str(e) or e.__class__.__name__
                if attempt >= max_attempts:
                    artifact_id, _artifact_dir = _write_compile_failure_artifacts(
                        cache_dir=cache_dir,
                        spec=spec,
                        verilog_text=verilog,
                        error_text=last_error,
                    )
                    raise PhyEngineError(
                        "Circuit compilation failed after multiple attempts. "
                        f"(artifact_id={artifact_id})\n"
                        f"Compiler output:\n{truncate(last_error, max_chars=2000)}"
                    ) from e
                verilog = llm_fix_verilog(
                    ollama=ollama,
                    spec=spec,
                    prior_verilog=verilog,
                    error_text=last_error,
                )

        artifact_dir = None
        artifact_verilog = None
        artifact_sav = None

        counts = None
        try:
            counts = load_plsav_counts(out_sav)
        except PlSavError:
            counts = None

        plsav_elements = counts.elements if counts is not None else None
        too_large = (
            isinstance(plsav_elements, int)
            and publish_max_elements > 0
            and plsav_elements > publish_max_elements
        )

        if too_large:
            artifact_dir, artifact_verilog, artifact_sav = _write_artifacts(
                cache_dir=cache_dir,
                verilog_text=verilog,
                sav_path=out_sav,
            )
            return CircuitBuildResult(
                published=False,
                artifact_dir=artifact_dir,
                artifact_verilog_path=artifact_verilog,
                artifact_sav_path=artifact_sav,
                publish_block_reason="too_large",
                plsav_elements=plsav_elements,
            )

        if keep_temp or (not enable_publish) or dry_run:
            artifact_dir, artifact_verilog, artifact_sav = _write_artifacts(
                cache_dir=cache_dir,
                verilog_text=verilog,
                sav_path=out_sav,
            )

        if not enable_publish or dry_run:
            return CircuitBuildResult(
                published=False,
                artifact_dir=artifact_dir,
                artifact_verilog_path=artifact_verilog,
                artifact_sav_path=artifact_sav,
                plsav_elements=plsav_elements,
            )

        info = upload_sav_as_experiment(
            user=user,
            sav_path=out_sav,
            title=title,
            introduction=introduction,
            cache_dir=cache_dir,
            category_value=publish_category_value,
            tags=publish_tags,
        )

        return CircuitBuildResult(
            published=True,
            summary_id=str(info.get("summary_id")),
            artifact_dir=artifact_dir,
            artifact_verilog_path=artifact_verilog,
            artifact_sav_path=artifact_sav,
            plsav_elements=plsav_elements,
        )


def safe_reply(text: str, *, max_chars: int) -> str:
    return truncate(text, max_chars=max_chars)
