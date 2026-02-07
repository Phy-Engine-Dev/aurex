from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass


class PhyEngineError(RuntimeError):
    pass


@dataclass(frozen=True)
class Verilog2PlSavOptions:
    top: str | None = None
    extra_args: list[str] | None = None


def _is_executable_file(path: str) -> bool:
    return os.path.isfile(path) and os.access(path, os.X_OK)


def _is_shared_lib_file(path: str) -> bool:
    return os.path.isfile(path)


def _find_verilog2plsav_in_build_dir(build_dir: str) -> str | None:
    candidates = [
        os.path.join(build_dir, "verilog2plsav"),
        os.path.join(build_dir, "Release", "verilog2plsav"),
        os.path.join(build_dir, "verilog2plsav.exe"),
        os.path.join(build_dir, "Release", "verilog2plsav.exe"),
    ]
    for c in candidates:
        if _is_executable_file(c):
            return c
    return None


def _find_phyengine_lib_in_build_dir(build_dir: str) -> str | None:
    candidates = [
        os.path.join(build_dir, "libphyengine.so"),
        os.path.join(build_dir, "libphyengine.dylib"),
        os.path.join(build_dir, "phyengine.dll"),
        os.path.join(build_dir, "Release", "libphyengine.so"),
        os.path.join(build_dir, "Release", "libphyengine.dylib"),
        os.path.join(build_dir, "Release", "phyengine.dll"),
    ]
    for c in candidates:
        if _is_shared_lib_file(c):
            return c
    return None


def _cmake_configure_and_build(
    *,
    source_dir: str,
    build_dir: str,
    build_type: str,
    targets: list[str],
    build_timeout_sec: int,
) -> None:
    if shutil.which("cmake") is None:
        raise PhyEngineError("cmake not found in PATH (required for auto_build)")

    os.makedirs(build_dir, exist_ok=True)

    configure_cmd = [
        "cmake",
        "-S",
        source_dir,
        "-B",
        build_dir,
        f"-DCMAKE_BUILD_TYPE={build_type}",
    ]
    build_cmd = ["cmake", "--build", build_dir]
    for t in targets:
        if t:
            build_cmd.extend(["--target", t])

    try:
        subprocess.run(configure_cmd, check=True, timeout=build_timeout_sec)
        subprocess.run(build_cmd, check=True, timeout=build_timeout_sec)
    except subprocess.TimeoutExpired as e:
        raise PhyEngineError(f"Phy-Engine build timed out: {e}") from e
    except subprocess.CalledProcessError as e:
        raise PhyEngineError(f"Phy-Engine build failed: {e}") from e


def prebuild_phy_engine(
    *,
    cmake_source_dir: str,
    cmake_build_dir: str,
    cmake_build_type: str,
    build_timeout_sec: int,
    targets: list[str] | None = None,
) -> None:
    source_dir = os.path.abspath(cmake_source_dir)
    build_dir = os.path.abspath(cmake_build_dir)
    _cmake_configure_and_build(
        source_dir=source_dir,
        build_dir=build_dir,
        build_type=cmake_build_type,
        targets=targets or ["verilog2plsav", "phyengine"],
        build_timeout_sec=build_timeout_sec,
    )


def ensure_verilog2plsav(
    *,
    verilog2plsav_path: str,
    auto_build: bool,
    cmake_source_dir: str,
    cmake_build_dir: str,
    cmake_build_type: str,
    build_timeout_sec: int,
) -> str:
    if verilog2plsav_path:
        path = os.path.abspath(verilog2plsav_path)
        if not _is_executable_file(path):
            raise PhyEngineError(f"verilog2plsav not found or not executable: {path}")
        return path

    existing = _find_verilog2plsav_in_build_dir(os.path.abspath(cmake_build_dir))
    if existing is not None:
        return existing

    if not auto_build:
        raise PhyEngineError(
            "verilog2plsav is not configured. Set phy_engine.verilog2plsav_path or enable phy_engine.auto_build."
        )

    source_dir = os.path.abspath(cmake_source_dir)
    build_dir = os.path.abspath(cmake_build_dir)
    _cmake_configure_and_build(
        source_dir=source_dir,
        build_dir=build_dir,
        build_type=cmake_build_type,
        targets=["verilog2plsav"],
        build_timeout_sec=build_timeout_sec,
    )

    built = _find_verilog2plsav_in_build_dir(build_dir)
    if built is None:
        raise PhyEngineError(
            f"verilog2plsav build succeeded but binary not found in {build_dir}"
        )
    return built


def ensure_phyengine_lib(
    *,
    phyengine_lib_path: str,
    auto_build: bool,
    cmake_source_dir: str,
    cmake_build_dir: str,
    cmake_build_type: str,
    build_timeout_sec: int,
) -> str:
    if phyengine_lib_path:
        path = os.path.abspath(phyengine_lib_path)
        if not _is_shared_lib_file(path):
            raise PhyEngineError(f"phyengine shared library not found: {path}")
        return path

    existing = _find_phyengine_lib_in_build_dir(os.path.abspath(cmake_build_dir))
    if existing is not None:
        return existing

    if not auto_build:
        raise PhyEngineError(
            "phyengine library is not configured. Set phy_engine.phyengine_lib_path or enable phy_engine.auto_build."
        )

    source_dir = os.path.abspath(cmake_source_dir)
    build_dir = os.path.abspath(cmake_build_dir)
    _cmake_configure_and_build(
        source_dir=source_dir,
        build_dir=build_dir,
        build_type=cmake_build_type,
        targets=["phyengine"],
        build_timeout_sec=build_timeout_sec,
    )

    built = _find_phyengine_lib_in_build_dir(build_dir)
    if built is None:
        raise PhyEngineError(
            f"phyengine build succeeded but shared library not found in {build_dir}"
        )
    return built


def verilog_to_plsav(
    *,
    verilog2plsav_bin: str,
    out_sav_path: str,
    in_verilog_path: str,
    options: Verilog2PlSavOptions | None = None,
    timeout_sec: int = 300,
) -> None:
    options = options or Verilog2PlSavOptions()
    cmd = [verilog2plsav_bin, out_sav_path, in_verilog_path]
    if options.top:
        cmd.extend(["--top", options.top])
    if options.extra_args:
        cmd.extend(list(options.extra_args))

    try:
        subprocess.run(cmd, check=True, timeout=timeout_sec, capture_output=True, text=True)
    except subprocess.TimeoutExpired as e:
        raise PhyEngineError(f"verilog2plsav timed out: {e}") from e
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or "").strip()
        if len(stderr) > 4000:
            stderr = stderr[-4000:]
        raise PhyEngineError(f"verilog2plsav failed (exit={e.returncode}): {stderr}") from e

    if not os.path.exists(out_sav_path):
        raise PhyEngineError(f"verilog2plsav did not produce output: {out_sav_path}")
