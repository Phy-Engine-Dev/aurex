from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass


class PhyEngineBuildError(RuntimeError):
    pass


@dataclass(frozen=True)
class PhyEngineArtifacts:
    build_dir: str
    verilog2plsav_path: str
    phyengine_lib_path: str
    circuit_view_path: str = ""


def _find_verilog2plsav(build_dir: str) -> str:
    for name in ("verilog2plsav", "verilog2plsav.exe"):
        p = os.path.join(build_dir, name)
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return ""


def _find_phyengine_lib(build_dir: str) -> str:
    candidates = [
        "libphyengine.so",
        "libphyengine.dylib",
        "phyengine.dll",
        "phyengine.so",
    ]
    for name in candidates:
        p = os.path.join(build_dir, name)
        if os.path.isfile(p):
            return p
    return ""


def ensure_built(
    *,
    source_dir: str,
    build_dir: str,
    build_type: str,
    timeout_sec: int,
) -> PhyEngineArtifacts:
    os.makedirs(build_dir, exist_ok=True)

    cfg_cmd = ["cmake", "-S", source_dir, "-B", build_dir, f"-DCMAKE_BUILD_TYPE={build_type}"]
    if not os.path.exists(os.path.join(build_dir, "CMakeCache.txt")):
        compiler = next((name for name in ("clang++-22", "clang++-21", "g++-15", "g++-16") if shutil.which(name)), None)
        if compiler:
            cfg_cmd.append(f"-DCMAKE_CXX_COMPILER={compiler}")
        cfg_cmd.append("-DPHY_ENGINE_USE_LEVELDB=OFF")
    try:
        subprocess.run(cfg_cmd, check=True, capture_output=True, text=True, timeout=timeout_sec)
    except subprocess.TimeoutExpired as e:
        raise PhyEngineBuildError(f"CMake configure timed out after {timeout_sec}s") from e
    except subprocess.CalledProcessError as e:
        raise PhyEngineBuildError(f"CMake configure failed: {e.stderr or e.stdout}") from e

    build_cmd = ["cmake", "--build", build_dir, "--target", "verilog2plsav", "phyengine", "circuit_view", "--parallel", "2"]
    try:
        subprocess.run(build_cmd, check=True, capture_output=True, text=True, timeout=timeout_sec)
    except subprocess.TimeoutExpired as e:
        raise PhyEngineBuildError(f"CMake build timed out after {timeout_sec}s") from e
    except subprocess.CalledProcessError as e:
        raise PhyEngineBuildError(f"CMake build failed: {e.stderr or e.stdout}") from e

    v2p = _find_verilog2plsav(build_dir)
    lib = _find_phyengine_lib(build_dir)
    if not v2p:
        raise PhyEngineBuildError("Build succeeded but verilog2plsav not found in build_dir")
    if not lib:
        raise PhyEngineBuildError("Build succeeded but phyengine shared library not found in build_dir")

    renderer = os.path.join(build_dir, "circuit_view.exe" if os.name == "nt" else "circuit_view")
    if not os.path.isfile(renderer):
        raise PhyEngineBuildError("Build succeeded but circuit_view not found in build_dir")
    return PhyEngineArtifacts(build_dir=build_dir, verilog2plsav_path=v2p, phyengine_lib_path=lib, circuit_view_path=renderer)
