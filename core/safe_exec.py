"""
core/safe_exec.py
Restricted execution sandbox for LLM-generated pandas code.

Blocks file I/O, network, subprocess, dunder access, and dangerous builtins.
Used by LearningAgent._query_data() to answer cell-level / aggregation
questions about the user's uploaded data file.

This is not a security boundary — a determined attacker who controls the LLM
output could potentially bypass it. The protection is best-effort: pattern
blocking + restricted builtins + timeout. Anything truly sensitive should not
be accessible from the namespace handed in.
"""
from __future__ import annotations

import re
import threading
from typing import Any, Dict, Tuple

# Patterns that disqualify generated code outright.
_FORBIDDEN_PATTERNS = [
    r"__import__",
    r"__builtins__",
    r"__class__", r"__base__", r"__subclasses__",
    r"__globals__", r"__bases__", r"__mro__",
    r"\bimport\s+os\b",
    r"\bimport\s+sys\b",
    r"\bimport\s+subprocess\b",
    r"\bimport\s+pathlib\b",
    r"\bimport\s+shutil\b",
    r"\bimport\s+socket\b",
    r"\bimport\s+requests\b",
    r"\bimport\s+urllib\b",
    r"\bfrom\s+os\b",
    r"\bfrom\s+sys\b",
    r"\bfrom\s+subprocess\b",
    r"\bfrom\s+pathlib\b",
    r"\bopen\s*\(",
    r"\beval\s*\(",
    r"\bexec\s*\(",
    r"\bcompile\s*\(",
    r"\bgetattr\s*\(",
    r"\bsetattr\s*\(",
    r"\bdelattr\s*\(",
    r"\bglobals\s*\(",
    r"\blocals\s*\(",
    r"\.system\s*\(",
    r"\.popen\s*\(",
    r"\.to_csv\s*\(",
    r"\.to_excel\s*\(",
    r"\.to_pickle\s*\(",
    r"\.to_parquet\s*\(",
    r"\.to_sql\s*\(",
]

# Tiny safe subset of Python builtins.
_SAFE_BUILTINS: Dict[str, Any] = {
    "len": len, "min": min, "max": max, "sum": sum, "abs": abs,
    "round": round, "sorted": sorted, "reversed": reversed,
    "list": list, "dict": dict, "set": set, "tuple": tuple,
    "str": str, "int": int, "float": float, "bool": bool,
    "range": range, "enumerate": enumerate, "zip": zip,
    "map": map, "filter": filter,
    "isinstance": isinstance, "issubclass": issubclass,
    "any": any, "all": all,
    "print": print,
    "True": True, "False": False, "None": None,
}


class UnsafeCodeError(ValueError):
    """Raised when LLM-generated code fails the pre-execution safety check."""


class CodeTimeoutError(Exception):
    """Raised when sandboxed code exceeds its time budget."""


def is_safe_code(code: str) -> Tuple[bool, str]:
    """Return (ok, reason). reason is the matching forbidden pattern."""
    for pat in _FORBIDDEN_PATTERNS:
        m = re.search(pat, code)
        if m:
            return False, f"forbidden pattern: {m.group(0)!r}"
    return True, ""


def safe_exec(
    code: str,
    namespace: Dict[str, Any],
    timeout: int = 15,
) -> Dict[str, Any]:
    """
    Execute `code` against `namespace`.

    `namespace` is mutated in place (and returned) so the caller can read
    out an `answer` variable or any other top-level binding.

    Implementation note: timeout uses a daemon worker thread + join(timeout).
    SIGALRM was tried previously but only works on the main thread, which
    breaks under FastAPI/Uvicorn worker threads. The trade-off here is that
    a runaway exec() leaks a daemon thread until process exit — acceptable
    since the code is pre-validated and the namespace contains no syscalls.

    Raises:
        UnsafeCodeError: if the code matches a forbidden pattern.
        CodeTimeoutError: if execution exceeds `timeout` seconds.
        Any exception raised by the code itself (NameError, KeyError, etc.).
    """
    ok, reason = is_safe_code(code)
    if not ok:
        raise UnsafeCodeError(f"Unsafe code rejected — {reason}")

    safe_globals: Dict[str, Any] = {"__builtins__": _SAFE_BUILTINS}
    safe_globals.update(namespace)

    captured: Dict[str, BaseException | None] = {"exc": None}

    def _run() -> None:
        try:
            exec(code, safe_globals, namespace)  # noqa: S102
        except BaseException as e:  # noqa: BLE001 — re-raised in caller
            captured["exc"] = e

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    worker.join(timeout)

    if worker.is_alive():
        # Thread is still running. We can't kill it cleanly in CPython, but it
        # will be reaped at process exit (daemon=True) and won't block shutdown.
        raise CodeTimeoutError(f"Code execution exceeded {timeout}s")

    if captured["exc"] is not None:
        raise captured["exc"]

    return namespace
