from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from dataclasses import dataclass

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PHY_LAB_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PHY_LAB_DIR not in sys.path:
    sys.path.insert(0, PHY_LAB_DIR)

from config import config_dir, load_config, pick_cache_dir  # noqa: E402
from ollama import OllamaClient  # noqa: E402
from plar import email_login  # noqa: E402
from agent import agent_mode_run  # noqa: E402


@dataclass(frozen=True)
class Case:
    name: str
    text: str
    max_seconds: int = 180
    require_hex_summary_id: bool = False
    require_artifacts: bool = False


_SCRIPT_CHECKS: dict[str, re.Pattern[str]] = {
    "cjk": re.compile(r"[\u4e00-\u9fff]"),
    "ja": re.compile(r"[\u3040-\u30ff]"),
    "ko": re.compile(r"[\uac00-\ud7af]"),
    "ru": re.compile(r"[\u0400-\u04ff]"),
    "ar": re.compile(r"[\u0600-\u06ff]"),
    "hi": re.compile(r"[\u0900-\u097f]"),
    "th": re.compile(r"[\u0e00-\u0e7f]"),
}


def _expected_script_for_case(name: str) -> str | None:
    if name.startswith("zh_"):
        return "cjk"
    if name.startswith("ja_"):
        return "ja"
    if name.startswith("ko_"):
        return "ko"
    if name.startswith("ru_"):
        return "ru"
    if name.startswith("ar_"):
        return "ar"
    if name.startswith("hi_"):
        return "hi"
    if name.startswith("th_"):
        return "th"
    return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Multilingual smoke test for phy_lab agent mode.")
    p.add_argument("--config", required=True, help="Path to config JSON")
    p.add_argument("--cases", default="default", help="default|all")
    p.add_argument("--max-seconds", default=None, type=int, help="Override max seconds per case")
    p.add_argument("--fail-fast", action="store_true", help="Stop on first failure")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    base_dir = config_dir(args.config)
    cache_dir = pick_cache_dir(args.config, config=cfg)
    os.makedirs(cache_dir, exist_ok=True)

    logger = logging.getLogger("multilang_smoketest")
    logger.setLevel(logging.ERROR)

    # Login once.
    if not getattr(cfg.account, "password", None):
        print("ERROR: account.password is required for this smoke test")
        return 2
    user = email_login(
        email=cfg.account.email,
        password=cfg.account.password,
        cache_dir=cache_dir,
        http_timeout_sec=60.0,
    )

    endpoints = list(getattr(cfg.ollama, "base_urls", None) or []) or [cfg.ollama.base_url]
    model = cfg.ollama.model
    timeout_sec = int(getattr(cfg.ollama, "request_timeout_sec", 240) or 240)
    temperature = float(getattr(cfg.ollama, "temperature", 0.2) or 0.2)
    num_predict = int(getattr(cfg.ollama, "num_predict", 2048) or 2048)
    gptoss_opt = bool(getattr(cfg.ollama, "gptoss_optimization", False))

    client: OllamaClient | None = None
    base_url = ""
    last_err: Exception | None = None
    for ep in [str(x or "").strip() for x in endpoints if str(x or "").strip()]:
        c = OllamaClient(
            base_url=ep,
            model=model,
            timeout_sec=timeout_sec,
            temperature=temperature,
            num_predict=num_predict,
            gptoss_optimization=gptoss_opt,
        )
        try:
            ping = c.chat(messages=[{"role": "user", "content": "Reply with exactly: OK"}])
            if (ping or "").strip() == "OK":
                client = c
                base_url = ep
                break
        except Exception as e:
            last_err = e
            continue
    if client is None:
        print("ERROR: could not reach any Ollama endpoint from config.")
        print(f"endpoints={endpoints!r} model={model!r} last_error={last_err}")
        return 3

    default_cases: list[Case] = [
        Case(
            "zh_user_pick",
            "请你告诉我 Neptumium 发布的最有趣的实验是什么？直接给 1 个，并附上 Category+SummaryID+Subject。",
            240,
            require_hex_summary_id=True,
        ),
        Case(
            "en_simulate",
            "simulate DC: V=5V, R1=100ohm, R2=200ohm in series. Report current and node voltages.",
            180,
        ),
        Case(
            "ja_simulate",
            "直列抵抗回路を simulate: V=9V, R1=1kΩ, R2=2kΩ。電流と各抵抗の電圧降下を教えて。",
            180,
        ),
        Case(
            "ko_simulate",
            "직렬 저항 회로 simulate: V=3.3V, R1=330ohm, R2=660ohm. 전류와 각 저항 전압강하를 알려줘.",
            180,
        ),
        Case(
            "fr_explain",
            "Explique en 5 phrases la constante de temps d’un circuit RC (τ=RC) et donne un exemple numérique.",
            120,
        ),
        Case(
            "es_circuit",
            "Por favor, implementa un full adder de 1 bit en Verilog y compílalo a .sav; no publiques.",
            240,
            require_artifacts=True,
        ),
        Case(
            "de_explain",
            "Erkläre kurz den Unterschied zwischen Spannungsteiler und Stromteiler und gib ein Zahlenbeispiel.",
            120,
        ),
        Case(
            "ru_explain",
            "Кратко объясни закон Кирхгофа для токов (KCL) на простом примере.",
            120,
        ),
        Case(
            "ar_explain",
            "اشرح قانون أوم مع مثال عددي بسيط (5V و 100Ω).",
            120,
        ),
        Case(
            "hi_explain",
            "RC लो-पास फ़िल्टर क्या है? cutoff frequency का सूत्र लिखो और एक उदाहरण दो।",
            120,
        ),
        Case(
            "th_explain",
            "อธิบายกฎของโอห์ม และยกตัวอย่างคำนวณแบบสั้น ๆ (5V, 100Ω)",
            120,
        ),
        Case(
            "vi_ohm",
            "Hãy giải thích định luật Ohm và cho ví dụ tính nhanh (5V, 100Ω).",
            120,
        ),
    ]

    all_extra: list[Case] = [
        Case(
            "zh_first_work",
            "请你告诉我紫兰斋的第一个作品是什么？只需要 Category+SummaryID+Subject。",
            180,
        ),
        Case(
            "ptbr_logic",
            "Por que portas NAND são funcionalmente completas? Dê um exemplo simples.",
            120,
        ),
        Case(
            "id_dcac",
            "Jelaskan perbedaan arus AC dan DC secara singkat, dan berikan contoh penggunaan.",
            120,
        ),
    ]

    cases = list(default_cases)
    if str(args.cases).strip().lower() == "all":
        cases.extend(all_extra)

    override = args.max_seconds
    if override is not None and override > 0:
        cases = [Case(c.name, c.text, int(override)) for c in cases]

    failures: list[str] = []
    print("MULTILANG AGENT SMOKE TEST")
    print(f"ollama={base_url} model={cfg.ollama.model}")
    print(f"cases={len(cases)}")
    print("-")

    for idx, c in enumerate(cases, start=1):
        out = agent_mode_run(
            ollama=client,
            user=user,
            cfg=cfg,
            cache_dir=cache_dir,
            config_base_dir=base_dir,
            dry_run=True,
            logger=logger,
            task=c.text,
            context_json=None,
            history=[],
            requester_nickname=getattr(user, "nickname", None),
            requester_user_id=getattr(user, "user_id", None),
            max_seconds=int(c.max_seconds),
            max_steps=12,
        )
        out = (out or "").strip()
        ok = bool(out) and (not out.startswith("ERROR:"))

        if ok and c.require_hex_summary_id:
            ok = bool(re.search(r"[0-9a-fA-F]{24}", out))
        if ok and c.require_artifacts:
            ok = ("artifact_sav_path" in out) and ("artifact_verilog_path" in out)

        script = _expected_script_for_case(c.name)
        script_ok = True
        if script is not None:
            script_ok = bool(_SCRIPT_CHECKS[script].search(out))

        status = "PASS" if (ok and script_ok) else "FAIL"
        preview = out.replace("\n", " ")
        if len(preview) > 160:
            preview = preview[:160] + "..."
        extra = f" script_mismatch({script})" if (script and not script_ok) else ""
        print(f"{status} {idx:02d}/{len(cases):02d} {c.name}{extra}: {preview}")

        if not (ok and script_ok):
            failures.append(c.name)
            if args.fail_fast:
                break

    print("-")
    if failures:
        print(f"fail={len(failures)}: {', '.join(failures)}")
        return 1
    print("pass=all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
