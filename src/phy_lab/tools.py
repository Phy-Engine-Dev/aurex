from __future__ import annotations

import hashlib
import html
import os
import re
import shutil
import tempfile
import time
import urllib.parse
import uuid
from dataclasses import dataclass
from typing import Any

import requests

from ollama import OllamaClient
from plsav import PlSavError, load_plsav_counts
from pe_sim import PhyEngineLib, SeriesVdcResistorsSpec
from phy_engine import (
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
        from physicsLab import Category
    except Exception as e:  # pragma: no cover
        raise PLARError(f"Failed to import physicsLab.Category: {e}") from e

    categories = [Category.Experiment, Category.Discussion]
    scanned: list[dict[str, Any]] = []

    for cat in categories:
        if len(scanned) >= max_scan:
            break
        page = query_experiments(user, category=cat, take=min(50, max_scan - len(scanned)))
        scanned.extend(page)

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
        if item_id:
            lines.append(f"{idx}. {subject} (ID: {item_id})")
        else:
            lines.append(f"{idx}. {subject}")
    return "\n".join(lines)


def llm_generate_verilog(*, ollama: OllamaClient, spec: str) -> str:
    prompt = (
        "Generate synthesizable Verilog-2001 code for the following specification.\n"
        "Rules:\n"
        "- Output ONLY a single fenced code block.\n"
        "- The fenced block language tag must be 'verilog'.\n"
        "- Include a clear top module.\n"
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
        "Rules:\n"
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
        proxies = None
        p = (proxy or "").strip()
        if p:
            if "://" not in p:
                p = "http://" + p
            if p.lower().startswith("socks"):
                raise RuntimeError(
                    "SOCKS proxy requires requests[socks]. Use an HTTP proxy URL or install requests[socks]."
                )
            proxies = {"http": p, "https": p}

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
            )
        }
        r = requests.get(url, headers=headers, timeout=float(timeout_sec), proxies=proxies)
        r.raise_for_status()
        cached = r.text
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
) -> str:
    query = (query or "").strip()
    if not query:
        return "Provide a query string."
    if max_results <= 0:
        max_results = 5
    if max_results > 10:
        max_results = 10

    url = "https://duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})

    cache_root = os.path.join(cache_dir, "web_cache", "duckduckgo")
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    cache_path = os.path.join(cache_root, f"{digest}.html")
    cached = _cache_get(cache_path, ttl_sec=int(ttl_sec))
    if cached is None:
        proxies = None
        p = (proxy or "").strip()
        if p:
            if "://" not in p:
                p = "http://" + p
            if p.lower().startswith("socks"):
                raise RuntimeError(
                    "SOCKS proxy requires requests[socks]. Use an HTTP proxy URL or install requests[socks]."
                )
            proxies = {"http": p, "https": p}
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
            )
        }
        r = requests.get(url, headers=headers, timeout=float(timeout_sec), proxies=proxies)
        r.raise_for_status()
        cached = r.text
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
        return f"No results parsed from DuckDuckGo.\nQuery: {query}"

    lines = ["DuckDuckGo results:"]
    for i, (t, u) in enumerate(results, start=1):
        lines.append(f"{i}. {truncate(t, max_chars=120)}")
        lines.append(f"   {u}")
    return "\n".join(lines)


def web_search(
    *,
    query: str,
    cache_dir: str,
    proxy: str = "",
    timeout_sec: int = 20,
    ttl_sec: int = 3600,
    max_results: int = 5,
    fallback_to_ddg: bool = True,
) -> str:
    res = web_search_google(
        query=query,
        cache_dir=cache_dir,
        proxy=proxy,
        timeout_sec=timeout_sec,
        ttl_sec=ttl_sec,
        max_results=max_results,
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
            )
        except Exception:
            return res
    return res


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
                    options=Verilog2PlSavOptions(top=None, extra_args=extra_args_list or None),
                    timeout_sec=int(getattr(phy_engine_cfg, "run_timeout_sec", 300)),
                )
                last_error = None
                break
            except Exception as e:
                last_error = str(e)
                if attempt >= max_attempts:
                    raise
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
            category_value="Experiment",
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
