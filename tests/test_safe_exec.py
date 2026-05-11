"""
tests/test_safe_exec.py
Tests for the safe-exec sandbox used by LearningAgent._query_data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.safe_exec import (
    CodeTimeoutError,
    UnsafeCodeError,
    is_safe_code,
    safe_exec,
)


class TestIsSafeCode:

    def test_simple_pandas_passes(self):
        ok, _ = is_safe_code("answer = df.shape[0]")
        assert ok

    def test_dunder_import_rejected(self):
        ok, reason = is_safe_code("__import__('os')")
        assert not ok
        assert "__import__" in reason

    def test_import_os_rejected(self):
        ok, _ = is_safe_code("import os\nanswer = os.getcwd()")
        assert not ok

    def test_import_subprocess_rejected(self):
        ok, _ = is_safe_code("import subprocess; subprocess.run(['ls'])")
        assert not ok

    def test_open_rejected(self):
        ok, _ = is_safe_code("answer = open('/etc/passwd').read()")
        assert not ok

    def test_eval_rejected(self):
        ok, _ = is_safe_code("answer = eval('1+1')")
        assert not ok

    def test_exec_rejected(self):
        ok, _ = is_safe_code("exec('answer = 1')")
        assert not ok

    def test_getattr_rejected(self):
        ok, _ = is_safe_code("answer = getattr(df, 'shape')")
        assert not ok

    def test_class_subclasses_rejected(self):
        ok, _ = is_safe_code("answer = ().__class__.__base__.__subclasses__()")
        assert not ok

    def test_to_csv_rejected(self):
        ok, _ = is_safe_code("df.to_csv('out.csv')")
        assert not ok

    def test_dot_system_rejected(self):
        ok, _ = is_safe_code("import os; os.system('rm -rf /')")
        assert not ok


class TestSafeExec:

    def test_scalar_answer(self):
        ns = {"df": pd.DataFrame({"a": [1, 2, 3]})}
        ns = safe_exec("answer = df['a'].sum()", ns)
        assert ns["answer"] == 6

    def test_dataframe_answer(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        ns = {"df": df, "pd": pd}
        ns = safe_exec("answer = df[df['a'] > 1]", ns)
        assert isinstance(ns["answer"], pd.DataFrame)
        assert len(ns["answer"]) == 2

    def test_list_answer(self):
        ns = {"sheets": {"A": pd.DataFrame(), "B": pd.DataFrame()}}
        ns = safe_exec("answer = list(sheets.keys())", ns)
        assert ns["answer"] == ["A", "B"]

    def test_pandas_methods_available(self):
        df = pd.DataFrame({"x": [10, 20, 30]})
        ns = {"df": df, "np": np}
        ns = safe_exec("answer = df['x'].mean()", ns)
        assert ns["answer"] == 20.0

    def test_unsafe_code_raises(self):
        with pytest.raises(UnsafeCodeError):
            safe_exec("__import__('os').getcwd()", {})

    def test_runtime_error_propagates(self):
        with pytest.raises(KeyError):
            safe_exec("answer = df['missing_col']",
                      {"df": pd.DataFrame({"a": [1]})})

    def test_no_globals_leak(self):
        """Code cannot access process-level builtins like __import__."""
        with pytest.raises(Exception):
            safe_exec("answer = __import__('os')", {})

    def test_timeout_triggers(self):
        # Thread-based timeout works on all platforms / all threads.
        with pytest.raises(CodeTimeoutError):
            safe_exec("while True: pass", {}, timeout=1)

    def test_timeout_works_in_worker_thread(self):
        """Regression: previously used signal.SIGALRM which only worked in main thread."""
        import threading
        results: dict = {}

        def _run_in_worker():
            try:
                safe_exec("answer = 1 + 1", {}, timeout=5)
                results["ok"] = True
            except Exception as exc:
                results["err"] = str(exc)

        t = threading.Thread(target=_run_in_worker)
        t.start()
        t.join(10)
        assert results.get("ok") is True, f"Worker-thread exec failed: {results.get('err')}"
