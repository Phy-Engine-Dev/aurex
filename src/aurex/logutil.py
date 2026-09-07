from __future__ import annotations

import logging
import os
import sys
from typing import Any


def truncate(text: Any, *, max_chars: int) -> str:
    s = str(text or "")
    if max_chars <= 0 or len(s) <= max_chars:
        return s
    return s[: max_chars - 1] + "…"


def parse_level(name: str) -> int:
    s = str(name or "").strip().upper()
    mapping = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "WARN": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
    }
    return mapping.get(s, logging.INFO)


def setup_logger(*, cache_dir: str, level: str, name: str = "aurex3") -> logging.Logger:
    os.makedirs(cache_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(parse_level(level))
    logger.propagate = False

    # Idempotent: avoid duplicate handlers if setup is called multiple times.
    if getattr(logger, "_aurex3_configured", False):
        return logger

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setLevel(parse_level(level))
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    log_path = os.path.join(cache_dir, "aurex3.log")
    try:
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(parse_level(level))
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except Exception:
        # If file logging fails, still keep stdout handler.
        pass

    setattr(logger, "_aurex3_configured", True)
    logger.debug("Logger configured (level=%s, file=%s)", level, log_path)
    return logger
