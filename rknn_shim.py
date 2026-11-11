"""Compatibility shims that let rknn-toolkit2 run on this aarch64 board.

``rknn-toolkit2`` is normally an x86_64 tool, but Rockchip also publishes
aarch64 wheels that contain the full compiler (``librknnc.so``), so the
ONNX -> RKNN conversion can be done locally - no separate x86 machine is
needed.

Two import time problems are worked around:

``torch``
    The compiled extension imports torch when the Python API is loaded,
    even though only the ONNX front end is used.  A stub package lives in
    ``.venv-rknn/lib/python3.11/site-packages/torch/__init__.py``.

``rknn.api.gdq_quant``
    Its Cython module needs a real torch while it is being initialised.
    GDQ (gradient based data free quantisation) is not used by the normal
    quantisation path, so the module is replaced by a stub.

Import this module **before** ``from rknn.api import RKNN``.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys
import types

_GDQ_EXPORTS = ("GDQQuantizer", "GdqQuant", "GdqQuantizer", "BaseQuantizer")


class _GdqStubLoader(importlib.abc.Loader):
    """Provides the names rknn.api.quantizer imports from gdq_quant."""

    def create_module(self, spec):
        module = types.ModuleType(spec.name)
        for name in _GDQ_EXPORTS:
            setattr(
                module,
                name,
                type(name, (), {"__init__": lambda self, *a, **k: None}),
            )
        return module

    def exec_module(self, module):
        return None


class _GdqStubFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "rknn.api.gdq_quant":
            return importlib.machinery.ModuleSpec(fullname, _GdqStubLoader())
        return None


def _patch_ctypes_for_runtime() -> None:
    """Point the toolkit's runtime loader at the system librknnrt.so.

    rknn-toolkit2 hard codes ``/usr/lib64/librknnrt.so``; on this board the
    library lives in ``/usr/lib`` (and /usr/lib64 cannot be created without
    root).  Only the optional ``accuracy_analysis`` helper needs it.
    """
    import ctypes

    if getattr(ctypes, "_rknn_patched", False):
        return
    original = ctypes.CDLL

    def _cdll(name, *args, **kwargs):
        if isinstance(name, str) and "librknnrt" in name:
            name = "/usr/lib/librknnrt.so"
        return original(name, *args, **kwargs)

    ctypes.CDLL = _cdll
    ctypes._rknn_patched = True


def install() -> None:
    """Register the shims (idempotent)."""
    if not any(isinstance(f, _GdqStubFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _GdqStubFinder())
    _patch_ctypes_for_runtime()


install()
