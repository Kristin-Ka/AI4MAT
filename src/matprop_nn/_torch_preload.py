"""Preload NVIDIA shared libraries that torch needs at runtime.

Pip-installed torch CUDA wheels expect ``libcusparseLt.so.0`` (and friends)
to be on ``LD_LIBRARY_PATH``.  The ``nvidia-cusparselt-cu12`` wheel ships
the library under ``site-packages/nvidia/cusparselt/lib/`` but does not
register it with the dynamic loader, so importing torch fails with
``ImportError: libcusparseLt.so.0: cannot open shared object file``.

This module locates the library inside any installed nvidia-* package and
preloads it via ``ctypes.CDLL`` before torch is imported.  It is a no-op
if the library is already on the path.
"""

from __future__ import annotations

import ctypes
import logging
import os
import site
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_PRELOAD_LIBS = [
    "libcusparseLt.so.0",
]


def _candidate_dirs() -> list[Path]:
    dirs: list[Path] = []
    for base in site.getsitepackages() + [site.getusersitepackages()]:
        nvidia_root = Path(base) / "nvidia"
        if nvidia_root.is_dir():
            for sub in nvidia_root.iterdir():
                lib_dir = sub / "lib"
                if lib_dir.is_dir():
                    dirs.append(lib_dir)
    for p in sys.path:
        nvidia_root = Path(p) / "nvidia"
        if nvidia_root.is_dir():
            for sub in nvidia_root.iterdir():
                lib_dir = sub / "lib"
                if lib_dir.is_dir() and lib_dir not in dirs:
                    dirs.append(lib_dir)
    return dirs


def preload_cuda_libs() -> None:
    """Preload missing NVIDIA shared libs via ctypes so torch can import."""
    if sys.platform != "linux":
        return

    for libname in _PRELOAD_LIBS:
        try:
            ctypes.CDLL(libname, mode=ctypes.RTLD_GLOBAL)
            continue
        except OSError:
            pass

        found = False
        for d in _candidate_dirs():
            candidate = d / libname
            if candidate.exists():
                try:
                    ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
                    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
                    if str(d) not in ld_path.split(":"):
                        os.environ["LD_LIBRARY_PATH"] = f"{d}:{ld_path}".rstrip(":")
                    found = True
                    break
                except OSError as exc:
                    logger.debug("Failed to preload %s: %s", candidate, exc)

        if not found:
            logger.warning(
                "Could not locate %s; torch import may fail. "
                "Install nvidia-cusparselt-cu12 or set LD_LIBRARY_PATH.",
                libname,
            )


preload_cuda_libs()
