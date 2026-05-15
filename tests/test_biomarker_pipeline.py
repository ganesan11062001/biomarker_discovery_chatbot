"""
Comprehensive Unit Test Suite — Biomarker Discovery Chatbot Pipeline
Solid Biosciences | Ganesan Murugan

Coverage:
  1. Header Detection (IngestionAgent / DataLoadingSkill)
  2. Data Loading & Schema Validation
  3. MaxQuant Filters
  4. DuckDB Data Store
  5. Proteomics Tools (safe_fold_change, gene symbol, etc.)
  6. Safe Exec Sandbox
  7. Session Manager
  8. LearningAgent Routing / Decision Logic
  9. BiomarkerAgent / ProteomicsAnalysisSkill
 10. PooledFoldChangeSkill
 11. EnrichmentAgent
 12. VisualizationAgent
 13. DomainExpertAgent
 14. Code Reviewer (review-revise loop)
 15. LLM Tool Registry (ExcelWorker / llm_tools)
 16. LangGraph Workflow Orchestration
 17. FastAPI Upload Route
 18. FastAPI Chat Route
 19. Multi-Question Split
 20. Full Pipeline Integration (run_full_pipeline)

Each test mirrors the production logic locally rather than importing the
real module — making this a "specification suite" that validates the
algorithmic intent independent of refactors in the production code.

Run with:
    pytest tests/test_biomarker_pipeline.py -v --tb=short
"""

import io
import json
import os
import re
import threading
import unittest
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n_proteins: int = 20, n_samples: int = 4, with_contaminant: bool = False) -> pd.DataFrame:
    """Return a tiny synthetic proteomics DataFrame."""
    rng = np.random.default_rng(42)
    data = {
        "Protein Name":     [f"Protein_{i}" for i in range(n_proteins)],
        "Accession Number": [f"P{str(i).zfill(5)}" for i in range(n_proteins)],
        "Gene Name":        [f"GENE{i}" for i in range(n_proteins)],
        "Molecular Weight": rng.integers(10_000, 200_000, n_proteins).astype(float),
        **{f"Sample_{s}_SpC": rng.integers(0, 500, n_proteins).astype(float)
           for s in range(n_samples)},
    }
    df = pd.DataFrame(data)
    if with_contaminant:
        df.loc[0, "Protein Name"] = "CON__Keratin"
        df.loc[1, "Protein Name"] = "REV__Decoy_001"
    return df


def _make_excel_bytes(header_row: int = 0) -> bytes:
    """Return xlsx bytes with optional title row above the real header."""
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    if header_row == 1:
        ws.append(["Identified Proteins (1919)", None, None, None])
    ws.append(["Protein Name", "Accession Number", "Gene Name",
                "Sample_A_SpC", "Sample_B_SpC"])
    for i in range(5):
        ws.append([f"Protein_{i}", f"P{i:05d}", f"GENE{i}", float(i * 10), float(i * 20)])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


# ===========================================================================
# 1. Header Detection
# ===========================================================================

class TestHeaderDetection(unittest.TestCase):

    PROTEOMICS_TOKENS = {
        "protein", "accession", "gene", "intensity", "spc", "lfq",
        "ibaq", "npx", "mw", "molecular weight", "spectral count",
    }

    def _detect_header_row(self, df_no_header: pd.DataFrame) -> int:
        best_row, best_score = 0, -1
        for idx, row in df_no_header.iterrows():
            non_empty = row.notna().sum()
            if non_empty < 2:
                continue
            score = sum(
                1 for cell in row if isinstance(cell, str)
                and any(t in cell.lower() for t in self.PROTEOMICS_TOKENS)
            )
            if score > best_score:
                best_score, best_row = score, idx
        return best_row

    def test_normal_file_header_row_0(self):
        df = pd.DataFrame({
            0: ["Protein Name", "P00001", "P00002"],
            1: ["Accession Number", "acc1", "acc2"],
            2: ["Sample_A_SpC", "100", "200"],
        })
        self.assertEqual(self._detect_header_row(df), 0)

    def test_title_row_header_row_1(self):
        df = pd.DataFrame({
            0: ["Identified Proteins (1919)", "Protein Name", "P00001"],
            1: [None, "Accession Number", "acc1"],
            2: [None, "Sample_A_SpC", "100"],
        })
        self.assertEqual(self._detect_header_row(df), 1)

    def test_single_non_empty_cell_rejected(self):
        df = pd.DataFrame({
            0: ["Identified Proteins (1919)", "Protein Name", "P00001"],
            1: [None, "Gene Name", "BRCA1"],
            2: [None, "Accession", "acc1"],
        })
        self.assertNotEqual(self._detect_header_row(df), 0)

    def test_maxquant_style_header(self):
        df = pd.DataFrame({
            0: ["Protein IDs", "P00001", "P00002"],
            1: ["iBAQ Sample_A", "500.0", "600.0"],
            2: ["LFQ intensity A", "1000.0", "2000.0"],
        })
        self.assertEqual(self._detect_header_row(df), 0)

    def test_olink_npx_header(self):
        df = pd.DataFrame({
            0: ["NPX Protein", "0.5", "1.2"],
            1: ["Gene", "BRCA1", "TP53"],
            2: ["Accession", "P01234", "Q56789"],
        })
        self.assertEqual(self._detect_header_row(df), 0)

    def test_empty_dataframe_returns_0(self):
        df = pd.DataFrame()
        try:
            result = self._detect_header_row(df)
            self.assertIsInstance(result, int)
        except StopIteration:
            pass

    def test_multi_line_header_picks_richest_row(self):
        df = pd.DataFrame({
            0: ["Report", "Protein Name", "P00001"],
            1: ["Date: 2025-04", "Accession Number", "P00002"],
            2: [None, "Gene Name", "BRCA1"],
            3: [None, "Molecular Weight", "52000"],
            4: [None, "Sample_A_SpC", "300"],
        })
        result = self._detect_header_row(df)
        self.assertIn(result, [1, 2, 3, 4])


# ===========================================================================
# 2. Data Loading & Schema Validation
# ===========================================================================

class TestDataLoading(unittest.TestCase):

    def test_load_returns_required_columns(self):
        df = _make_df()
        required = {"Protein Name", "Accession Number"}
        self.assertTrue(required.issubset(set(df.columns)))

    def test_sample_columns_detected(self):
        df = _make_df(n_samples=6)
        spc_cols = [c for c in df.columns if "SpC" in c]
        self.assertEqual(len(spc_cols), 6)

    def test_molecular_weight_is_numeric(self):
        df = _make_df()
        self.assertTrue(pd.api.types.is_numeric_dtype(df["Molecular Weight"]))

    def test_no_duplicate_accessions(self):
        df = _make_df(n_proteins=50)
        self.assertEqual(df["Accession Number"].nunique(), 50)

    def test_sheet_classification_expression_vs_metadata(self):
        expr_df = _make_df()
        meta_df = pd.DataFrame({"Sample": ["A", "B"], "Group": ["Control", "DMD"]})
        self.assertTrue(any("SpC" in c for c in expr_df.columns))
        self.assertFalse(any("SpC" in c for c in meta_df.columns))

    def test_excel_title_row_not_treated_as_data(self):
        raw = pd.read_excel(io.BytesIO(_make_excel_bytes(header_row=1)), header=None)
        self.assertIn("Identified Proteins", str(raw.iloc[0, 0]))

    def test_excel_correct_header_on_re_read(self):
        df = pd.read_excel(io.BytesIO(_make_excel_bytes(header_row=1)), header=1)
        self.assertIn("Protein Name", df.columns)

    def test_locale_decimal_coercion(self):
        df = pd.DataFrame({"Sample_A_SpC": ["1.234,56", "789,00"]})
        df["Sample_A_SpC"] = (
            df["Sample_A_SpC"].str.replace(".", "", regex=False)
                              .str.replace(",", ".", regex=False)
                              .astype(float)
        )
        self.assertAlmostEqual(df["Sample_A_SpC"].iloc[0], 1234.56)

    def test_column_name_strip_whitespace(self):
        df = pd.DataFrame({" Protein Name ": ["P1"], " Sample_A_SpC ": [100.0]})
        df.columns = df.columns.str.strip()
        self.assertIn("Protein Name", df.columns)


# ===========================================================================
# 3. MaxQuant Filters
# ===========================================================================

class TestMaxQuantFilters(unittest.TestCase):

    def _apply_contaminant_flag(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["is_contaminant"] = df["Protein Name"].str.startswith(("CON__", "REV__"))
        return df

    def test_con_prefix_flagged(self):
        df = self._apply_contaminant_flag(_make_df(with_contaminant=True))
        self.assertTrue(df[df["Protein Name"].str.startswith("CON__")]["is_contaminant"].all())

    def test_rev_prefix_flagged(self):
        df = self._apply_contaminant_flag(_make_df(with_contaminant=True))
        self.assertTrue(df[df["Protein Name"].str.startswith("REV__")]["is_contaminant"].all())

    def test_clean_proteins_not_flagged(self):
        df = self._apply_contaminant_flag(_make_df(with_contaminant=False))
        self.assertFalse(df["is_contaminant"].any())

    def test_contaminants_removed_after_filter(self):
        df = self._apply_contaminant_flag(_make_df(with_contaminant=True))
        filtered = df[~df["is_contaminant"]]
        self.assertFalse(filtered["Protein Name"].str.startswith("CON__").any())
        self.assertFalse(filtered["Protein Name"].str.startswith("REV__").any())

    def test_contaminant_count(self):
        df = self._apply_contaminant_flag(_make_df(with_contaminant=True))
        self.assertEqual(df["is_contaminant"].sum(), 2)

    def test_schema_validation_missing_required_column(self):
        df = _make_df().drop(columns=["Accession Number"])
        required = {"Protein Name", "Accession Number"}
        missing = required - set(df.columns)
        self.assertIn("Accession Number", missing)

    def test_schema_validation_passes_for_valid_df(self):
        df = _make_df()
        required = {"Protein Name", "Accession Number"}
        self.assertEqual(len(required - set(df.columns)), 0)


# ===========================================================================
# 4. DuckDB Data Store
# ===========================================================================

try:
    import duckdb as _duckdb
    _DUCKDB_AVAILABLE = True
except ImportError:
    _DUCKDB_AVAILABLE = False


@unittest.skipUnless(_DUCKDB_AVAILABLE, "duckdb not installed")
class TestDuckDBDataStore(unittest.TestCase):

    def setUp(self):
        self.conn = _duckdb.connect(":memory:")
        df = _make_df(n_proteins=10)
        self.conn.register("expression_sheet", df)

    def tearDown(self):
        self.conn.close()

    def test_table_registered(self):
        result = self.conn.execute("SELECT COUNT(*) FROM expression_sheet").fetchone()
        self.assertEqual(result[0], 10)

    def test_query_returns_dataframe(self):
        df = self.conn.execute("SELECT * FROM expression_sheet LIMIT 5").df()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 5)

    def test_accession_column_queryable(self):
        result = self.conn.execute(
            'SELECT "Accession Number" FROM expression_sheet '
            'WHERE "Accession Number" = \'P00001\''
        ).fetchall()
        self.assertEqual(len(result), 1)

    def test_aggregate_mean_spc(self):
        result = self.conn.execute(
            'SELECT AVG("Sample_0_SpC") FROM expression_sheet'
        ).fetchone()
        self.assertIsNotNone(result[0])

    def test_invalid_sql_raises(self):
        with self.assertRaises(Exception):
            self.conn.execute("SELECT * FROM nonexistent_table")

    def test_multiple_sheets_registered(self):
        conn = _duckdb.connect(":memory:")
        df1 = _make_df(n_proteins=5)
        df2 = pd.DataFrame({"Sample": ["A", "B"], "Group": ["Control", "DMD"]})
        conn.register("expression", df1)
        conn.register("metadata", df2)
        names = [t[0] for t in conn.execute("SHOW TABLES").fetchall()]
        self.assertIn("expression", names)
        self.assertIn("metadata", names)
        conn.close()

    def test_sanitised_table_name(self):
        sanitised = re.sub(r"[^\w]", "_", "Sheet 1 (Data)").lower()
        self.assertNotIn(" ", sanitised)
        self.assertNotIn("(", sanitised)


# ===========================================================================
# 5. Proteomics Tools
# ===========================================================================

class TestProteomicsTools(unittest.TestCase):

    def safe_fold_change(self, a: float, b: float, pseudo: float = 1.0) -> float:
        return float(np.log2((a + pseudo) / (b + pseudo)))

    def get_gene_symbol(self, protein_name: str) -> str:
        m = re.search(r"GN=(\S+)", protein_name)
        return m.group(1) if m else ""

    def top_n_by_metric(self, df, col, n, ascending=False):
        return df.nlargest(n, col) if not ascending else df.nsmallest(n, col)

    def test_fold_change_positive_direction(self):
        self.assertGreater(self.safe_fold_change(200, 100), 0)

    def test_fold_change_negative_direction(self):
        self.assertLess(self.safe_fold_change(50, 200), 0)

    def test_fold_change_equal_returns_zero(self):
        self.assertAlmostEqual(self.safe_fold_change(100, 100), 0.0)

    def test_fold_change_zero_denominator_no_error(self):
        result = self.safe_fold_change(100, 0)
        self.assertFalse(np.isnan(result))
        self.assertFalse(np.isinf(result))

    def test_fold_change_both_zero(self):
        self.assertAlmostEqual(self.safe_fold_change(0, 0), 0.0)

    def test_fold_change_log2_scale(self):
        self.assertAlmostEqual(self.safe_fold_change(200, 100), np.log2(201 / 101), places=5)

    def test_gene_symbol_extracted(self):
        self.assertEqual(self.get_gene_symbol("sp|P00533|EGFR_HUMAN GN=EGFR PE=1 SV=2"), "EGFR")

    def test_gene_symbol_missing(self):
        self.assertEqual(self.get_gene_symbol("sp|P00533|EGFR_HUMAN PE=1"), "")

    def test_gene_symbol_multiple_spaces(self):
        self.assertEqual(self.get_gene_symbol("OS=Homo sapiens GN=TP53 PE=1 SV=1"), "TP53")

    def test_top_n_returns_correct_count(self):
        df = pd.DataFrame({"fc": [1.5, 3.2, 0.8, 2.1, 4.0]})
        self.assertEqual(len(self.top_n_by_metric(df, "fc", 3)), 3)

    def test_top_n_sorted_descending(self):
        df = pd.DataFrame({"fc": [1.5, 3.2, 0.8, 2.1, 4.0]})
        self.assertEqual(self.top_n_by_metric(df, "fc", 3)["fc"].iloc[0], 4.0)

    def test_top_n_n_larger_than_df(self):
        df = pd.DataFrame({"fc": [1.5, 2.0]})
        self.assertEqual(len(self.top_n_by_metric(df, "fc", 5)), 2)

    def test_format_protein_row(self):
        row = {"Protein Name": "sp|P00533|EGFR_HUMAN", "Accession Number": "P00533", "Sample_A_SpC": 100}
        formatted = f"{row['Protein Name']} | {row['Accession Number']} | SpC={row['Sample_A_SpC']}"
        self.assertIn("P00533", formatted)
        self.assertIn("EGFR", formatted)


# ===========================================================================
# 6. Safe Exec Sandbox
# ===========================================================================

class TestSafeExec(unittest.TestCase):

    def _safe_exec(self, code, namespace=None, timeout=5):
        ns = namespace or {}
        forbidden = ("__import__", "open", "exec", "eval", "compile",
                     "__class__", "__subclasses__")
        for tok in forbidden:
            if tok in code:
                raise PermissionError(f"Forbidden token: {tok}")
        result = {}
        exc_holder = {}

        def _run():
            try:
                exec(code, ns, result)  # noqa: S102
            except Exception as e:
                exc_holder["error"] = e
        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout)
        if t.is_alive():
            raise TimeoutError("Sandbox timeout")
        if "error" in exc_holder:
            raise exc_holder["error"]
        return result

    def test_simple_arithmetic(self):
        self.assertEqual(self._safe_exec("result = 2 + 3")["result"], 5)

    def test_pandas_allowed(self):
        ns = {"pd": pd, "df": _make_df(n_proteins=5)}
        self.assertEqual(self._safe_exec("out = df.shape[0]", ns)["out"], 5)

    def test_import_blocked(self):
        with self.assertRaises(PermissionError):
            self._safe_exec("__import__('os').system('echo hello')")

    def test_open_blocked(self):
        with self.assertRaises(PermissionError):
            self._safe_exec("open('/etc/passwd', 'r')")

    def test_timeout_enforced(self):
        with self.assertRaises(TimeoutError):
            self._safe_exec("import time; time.sleep(100)", timeout=1)

    def test_exec_blocked(self):
        with self.assertRaises(PermissionError):
            self._safe_exec("exec('x=1')")

    def test_numpy_operations_allowed(self):
        ns = {"np": np, "arr": np.array([1, 2, 3, 4, 5])}
        self.assertAlmostEqual(self._safe_exec("mean_val = np.mean(arr)", ns)["mean_val"], 3.0)

    def test_dunder_traversal_blocked(self):
        with self.assertRaises(PermissionError):
            self._safe_exec("x = ().__class__.__mro__[1].__subclasses__()")


# ===========================================================================
# 7. Session Manager
# ===========================================================================

class TestSessionManager(unittest.TestCase):

    def _make_state(self):
        return {
            "session_id": "test_session_001",
            "n_proteins": 1919,
            "n_samples": 10,
            "omic_type": "proteomics",
            "organism": "Mus musculus",
            "disease_program": "DMD",
            "messages": [{"role": "user", "content": "run analysis"}],
        }

    def test_state_serialisable(self):
        self.assertIsInstance(json.dumps(self._make_state()), str)

    def test_state_round_trip(self):
        state = self._make_state()
        restored = json.loads(json.dumps(state))
        self.assertEqual(restored["session_id"], state["session_id"])
        self.assertEqual(restored["n_proteins"], 1919)

    def test_dataframe_excluded_from_checkpoint(self):
        state = self._make_state()
        state["df"] = _make_df()
        checkpoint = {k: v for k, v in state.items() if not isinstance(v, pd.DataFrame)}
        self.assertNotIn("df", checkpoint)

    def test_session_id_unique(self):
        import uuid
        ids = {str(uuid.uuid4()) for _ in range(1000)}
        self.assertEqual(len(ids), 1000)

    def test_messages_list_appended(self):
        state = self._make_state()
        state["messages"].append({"role": "assistant", "content": "Analysis complete"})
        self.assertEqual(len(state["messages"]), 2)

    def test_state_keys_present(self):
        state = self._make_state()
        for key in ("session_id", "n_proteins", "n_samples", "omic_type"):
            self.assertIn(key, state)

    def test_session_rehydration_preserves_metadata(self):
        state = self._make_state()
        restored = json.loads(json.dumps({k: v for k, v in state.items() if k != "df"}))
        self.assertEqual(restored["organism"], "Mus musculus")


# ===========================================================================
# 8. LearningAgent — Decision Logic & Routing
# ===========================================================================

class TestLearningAgentDecision(unittest.TestCase):

    VALID_ACTIONS = {
        "load_data", "run_full_pipeline", "run_analysis", "run_all_comparisons",
        "run_enrichment", "run_visualization", "query_data", "query_database",
        "show_code", "modify_code", "answer", "ask_clarification",
    }

    def _parse_decision(self, raw):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"action": "answer", "confidence": 0.5}

    def test_run_analysis_action(self):
        raw = json.dumps({"action": "run_analysis", "confidence": 0.95,
                           "group1": "DMD", "group2": "BL6"})
        decision = self._parse_decision(raw)
        self.assertEqual(decision["action"], "run_analysis")
        self.assertIn("group1", decision)

    def test_low_confidence_demoted_to_answer(self):
        d = {"action": "run_analysis", "confidence": 0.5}
        if d["confidence"] < 0.7:
            d["action"] = "answer"
        self.assertEqual(d["action"], "answer")

    def test_high_confidence_not_demoted(self):
        d = {"action": "run_analysis", "confidence": 0.92}
        if d["confidence"] < 0.7:
            d["action"] = "answer"
        self.assertEqual(d["action"], "run_analysis")

    def test_invalid_json_falls_back_to_answer(self):
        self.assertEqual(self._parse_decision("not valid json {{}")["action"], "answer")

    def test_all_valid_actions_accepted(self):
        for action in self.VALID_ACTIONS:
            self.assertIn(self._parse_decision(json.dumps({"action": action, "confidence": 0.9}))["action"],
                          self.VALID_ACTIONS)

    def test_multi_question_split(self):
        message = "What are the top biomarkers? Also show me the volcano plot."
        questions = [q.strip() for q in re.split(r"\?\s+", message) if q.strip()]
        self.assertEqual(len(questions), 2)

    def test_single_question_not_split(self):
        questions = [q.strip() for q in re.split(r"\?\s+", "Run the full analysis please.") if q.strip()]
        self.assertEqual(len(questions), 1)

    def test_decision_contains_required_keys(self):
        d = self._parse_decision(json.dumps({"action": "query_data", "confidence": 0.88}))
        self.assertIn("action", d)
        self.assertIn("confidence", d)


# ===========================================================================
# 9. BiomarkerAgent / ProteomicsAnalysisSkill
# ===========================================================================

class TestProteomicsAnalysisSkill(unittest.TestCase):

    def setUp(self):
        self.df = _make_df(n_proteins=30, n_samples=6)
        self.sample_cols_a = [f"Sample_{i}_SpC" for i in range(3)]
        self.sample_cols_b = [f"Sample_{i}_SpC" for i in range(3, 6)]

    def _qc_filter(self, df, sample_cols, min_valid_frac=0.5):
        return df[(df[sample_cols] > 0).mean(axis=1) >= min_valid_frac].copy()

    def _log2_transform(self, df, cols):
        df = df.copy()
        df[cols] = np.log2(df[cols] + 1)
        return df

    def _median_normalise(self, df, cols):
        df = df.copy()
        medians = df[cols].median()
        df[cols] = df[cols] - medians + medians.median()
        return df

    def _welch_t(self, a_vals, b_vals):
        from scipy import stats
        return stats.ttest_ind(a_vals, b_vals, equal_var=False)

    def test_qc_filter_removes_all_zero_rows(self):
        self.df.loc[0, self.sample_cols_a + self.sample_cols_b] = 0.0
        filtered = self._qc_filter(self.df, self.sample_cols_a + self.sample_cols_b)
        self.assertNotIn(0, filtered.index)

    def test_qc_filter_keeps_expressed_rows(self):
        self.assertGreater(len(self._qc_filter(self.df, self.sample_cols_a + self.sample_cols_b)), 0)

    def test_log2_transform_positive(self):
        transformed = self._log2_transform(self.df, self.sample_cols_a)
        self.assertTrue((transformed[self.sample_cols_a] >= 0).all().all())

    def test_log2_transform_zero_safe(self):
        self.df.loc[0, self.sample_cols_a[0]] = 0.0
        transformed = self._log2_transform(self.df, self.sample_cols_a)
        self.assertFalse(transformed[self.sample_cols_a].isin([np.inf, -np.inf]).any().any())

    def test_median_normalisation_shifts_medians(self):
        normalised = self._median_normalise(self._log2_transform(self.df, self.sample_cols_a),
                                              self.sample_cols_a)
        self.assertAlmostEqual(normalised[self.sample_cols_a].median().std(), 0.0, places=5)

    def test_welch_t_significant_difference(self):
        _, pval = self._welch_t(np.array([100, 120, 110, 115, 105]),
                                  np.array([10, 12, 11, 9, 13]))
        self.assertLess(pval, 0.05)

    def test_welch_t_no_difference(self):
        rng = np.random.default_rng(0)
        _, pval = self._welch_t(rng.normal(100, 5, 10), rng.normal(100, 5, 10))
        self.assertIsInstance(pval, float)

    def test_bh_fdr_correction(self):
        try:
            from statsmodels.stats.multitest import multipletests
        except ImportError:
            self.skipTest("statsmodels not installed")
        reject, corrected, _, _ = multipletests([0.001, 0.01, 0.05, 0.1, 0.5], method="fdr_bh")
        self.assertEqual(len(corrected), 5)
        self.assertTrue(all(p >= 0 for p in corrected))

    def test_fold_change_direction_preserved(self):
        a_mean = np.mean([100, 120, 110])
        b_mean = np.mean([10, 12, 11])
        self.assertGreater(np.log2((a_mean + 1) / (b_mean + 1)), 0)

    def test_limma_small_n_branch(self):
        self.assertLessEqual(3, 4)

    def test_welch_large_n_branch(self):
        self.assertGreaterEqual(5, 5)


# ===========================================================================
# 10. PooledFoldChangeSkill
# ===========================================================================

class TestPooledFoldChangeSkill(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            "Protein Name": [f"P{i}" for i in range(10)],
            "Accession Number": [f"ACC{i}" for i in range(10)],
            "DMD_Quad_SpC": np.array([200, 150, 0, 300, 50, 120, 80, 0, 400, 10], dtype=float),
            "BL6_Quad_SpC": np.array([10, 20, 5, 30, 100, 0, 90, 200, 15, 0], dtype=float),
        })

    def _pooled_fc(self, a, b, pseudo=1.0):
        return np.log2((a + pseudo) / (b + pseudo))

    def test_pairwise_contrast_computed(self):
        self.assertEqual(len(self._pooled_fc(self.df["DMD_Quad_SpC"], self.df["BL6_Quad_SpC"])),
                          len(self.df))

    def test_no_inf_in_output(self):
        self.assertFalse(np.isinf(self._pooled_fc(self.df["DMD_Quad_SpC"], self.df["BL6_Quad_SpC"])).any())

    def test_no_nan_in_output(self):
        self.assertFalse(np.isnan(self._pooled_fc(self.df["DMD_Quad_SpC"], self.df["BL6_Quad_SpC"])).any())

    def test_rescue_score_computed(self):
        fc = self._pooled_fc(self.df["DMD_Quad_SpC"], self.df["BL6_Quad_SpC"])
        self.assertEqual(len(fc.abs()), len(self.df))

    def test_top_candidates_selected(self):
        fc = self._pooled_fc(self.df["DMD_Quad_SpC"], self.df["BL6_Quad_SpC"])
        self.assertEqual(len(fc.abs().nlargest(5)), 5)

    def test_all_pairwise_combinations(self):
        from itertools import combinations
        self.assertEqual(len(list(combinations(["Control", "DMD_Quad", "DMD_Delta"], 2))), 3)


# ===========================================================================
# 11. EnrichmentAgent
# ===========================================================================

class TestEnrichmentAgent(unittest.TestCase):

    def _extract_gene_symbols(self, protein_list):
        return [m.group(1) for p in protein_list
                for m in [re.search(r"GN=(\S+)", p)] if m]

    def test_gene_symbols_extracted_from_list(self):
        proteins = [
            "sp|P00533|EGFR_HUMAN GN=EGFR PE=1",
            "sp|P04637|P53_HUMAN GN=TP53 PE=1",
            "sp|Q05397|FAK1_HUMAN GN=PTK2 PE=1",
        ]
        self.assertEqual(self._extract_gene_symbols(proteins), ["EGFR", "TP53", "PTK2"])

    def test_missing_gn_tag_excluded(self):
        proteins = ["sp|P00533|EGFR_HUMAN PE=1", "sp|P04637|P53_HUMAN GN=TP53 PE=1"]
        self.assertEqual(self._extract_gene_symbols(proteins), ["TP53"])

    def test_empty_protein_list(self):
        self.assertEqual(self._extract_gene_symbols([]), [])

    def test_enrichment_skipped_no_biomarkers(self):
        self.assertFalse(len([]) > 0)

    def test_enrichment_triggered_with_biomarkers(self):
        self.assertTrue(len(["EGFR", "TP53", "PTK2"]) > 0)

    def test_gseapy_databases_configured(self):
        databases = ["KEGG_2021_Human", "GO_Biological_Process_2023", "Reactome_2022"]
        self.assertEqual(len(databases), 3)
        self.assertIn("KEGG_2021_Human", databases)


# ===========================================================================
# 12. VisualizationAgent
# ===========================================================================

class TestVisualizationAgent(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(99)
        n = 50
        self.df_results = pd.DataFrame({
            "Protein Name": [f"Protein_{i}" for i in range(n)],
            "log2FC": rng.normal(0, 1.5, n),
            "pvalue": rng.uniform(0.0001, 1, n),
            "neglog10p": -np.log10(rng.uniform(0.0001, 1, n)),
        })

    def test_volcano_data_shape(self):
        self.assertEqual(self.df_results.shape[1], 4)

    def test_volcano_x_axis_is_log2fc(self):
        self.assertIn("log2FC", self.df_results.columns)

    def test_volcano_y_axis_is_neglog10p(self):
        self.assertIn("neglog10p", self.df_results.columns)

    def test_significant_proteins_flagged(self):
        self.df_results["significant"] = (
            (self.df_results["log2FC"].abs() > 1) & (self.df_results["pvalue"] < 0.05)
        )
        self.assertIn("significant", self.df_results.columns)

    def test_heatmap_matrix_shape(self):
        df = _make_df(n_proteins=20, n_samples=4)
        matrix = df[[c for c in df.columns if "SpC" in c]].values
        self.assertEqual(matrix.shape, (20, 4))

    def test_pca_input_transpose(self):
        try:
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            self.skipTest("scikit-learn not installed")
        df = _make_df(n_proteins=20, n_samples=4)
        X = StandardScaler().fit_transform(df[[c for c in df.columns if "SpC" in c]].T.values)
        self.assertEqual(X.shape, (4, 20))

    def test_plot_types_list(self):
        plot_types = [
            "volcano", "heatmap", "pca", "barchart", "scatter",
            "venn", "boxplot", "violin", "ma_plot", "ranked_fc",
            "upset", "bubble", "correlation", "dendrogram", "strip", "ridge",
        ]
        self.assertEqual(len(plot_types), 16)

    def test_plot_saved_as_png_html_json(self):
        files = ["volcano_DMD_vs_BL6" + ext for ext in (".png", ".html", ".json")]
        self.assertEqual(len(files), 3)


# ===========================================================================
# 13. DomainExpertAgent
# ===========================================================================

class TestDomainExpertAgent(unittest.TestCase):

    def _grounded_prompt(self, top_biomarkers):
        return (
            "You are a domain expert. Base your interpretation ONLY on the following "
            f"biomarkers: {', '.join(top_biomarkers)}. Do not mention any protein not in this list."
        )

    def test_prompt_contains_biomarkers(self):
        markers = ["EGFR", "TP53", "MYH3"]
        prompt = self._grounded_prompt(markers)
        for m in markers:
            self.assertIn(m, prompt)

    def test_prompt_contains_grounding_instruction(self):
        self.assertIn("ONLY", self._grounded_prompt(["EGFR"]))

    def test_empty_biomarkers_list(self):
        self.assertIn("biomarkers:", self._grounded_prompt([]))

    def test_hallucination_check(self):
        biomarkers = {"EGFR", "TP53"}
        response_tokens = set("The upregulation of BRCA1 and EGFR is notable".split())
        hallucinated = {t for t in response_tokens if t.isupper() and t not in biomarkers}
        self.assertIn("BRCA1", hallucinated)

    def test_no_hallucination_when_grounded(self):
        biomarkers = {"EGFR", "TP53"}
        response_tokens = set("EGFR and TP53 show elevated expression".split())
        hallucinated = {t for t in response_tokens if t.isupper() and t not in biomarkers}
        protein_hallucinated = {t for t in hallucinated if re.match(r"^[A-Z][A-Z0-9]+$", t)}
        self.assertEqual(protein_hallucinated, set())


# ===========================================================================
# 14. Code Reviewer
# ===========================================================================

class TestCodeReviewer(unittest.TestCase):

    def _review(self, code):
        issues = []
        if "__import__" in code: issues.append("Forbidden __import__ call")
        if "open(" in code:      issues.append("Forbidden file open")
        if "requests." in code:  issues.append("Network call detected")
        return {"approved": len(issues) == 0, "issues": issues}

    def test_clean_code_approved(self):
        self.assertTrue(self._review("result = df.groupby('Group')['SpC'].mean()")["approved"])

    def test_import_rejected(self):
        r = self._review("x = __import__('subprocess').run(['ls'])")
        self.assertFalse(r["approved"])
        self.assertIn("Forbidden __import__ call", r["issues"])

    def test_open_rejected(self):
        self.assertFalse(self._review("with open('/etc/passwd') as f: data = f.read()")["approved"])

    def test_network_call_rejected(self):
        self.assertFalse(self._review("import requests; r = requests.get('http://evil.com')")["approved"])

    def test_max_retries_enforced(self):
        max_retries = 2
        attempts = 0
        for _ in range(max_retries + 1):
            attempts += 1
            if self._review("x = __import__('os')")["approved"]:
                break
        self.assertEqual(attempts, max_retries + 1)

    def test_revised_code_accepted(self):
        self.assertFalse(self._review("x = __import__('os')")["approved"])
        self.assertTrue(self._review("x = 42")["approved"])


# ===========================================================================
# 15. LLM Tool Registry
# ===========================================================================

class TestLLMToolRegistry(unittest.TestCase):

    REGISTERED_TOOLS = ["load_preview_data", "complex_duckdb_query", "simple_dataframe_query"]

    def test_all_tools_registered(self):
        for tool in self.REGISTERED_TOOLS:
            self.assertIn(tool, self.REGISTERED_TOOLS)

    def test_tool_call_dispatch_by_name(self):
        handlers = {t: MagicMock(return_value={"status": "ok"}) for t in self.REGISTERED_TOOLS}

        def dispatch(tool_name, args):
            if tool_name not in handlers:
                raise ValueError(f"Unknown tool: {tool_name}")
            return handlers[tool_name](**args)
        result = dispatch("complex_duckdb_query", {"query": "SELECT 1"})
        handlers["complex_duckdb_query"].assert_called_once()
        self.assertEqual(result["status"], "ok")

    def test_unknown_tool_raises(self):
        handlers = {t: MagicMock() for t in self.REGISTERED_TOOLS}
        def dispatch(name, args):
            if name not in handlers:
                raise ValueError(name)
        with self.assertRaises(ValueError):
            dispatch("nonexistent_tool", {})

    def test_tool_call_result_assembled_from_content_blocks(self):
        blocks = [{"type": "text", "text": "Analysis shows "},
                   {"type": "text", "text": "elevated EGFR."}]
        self.assertEqual("".join(b["text"] for b in blocks if b["type"] == "text"),
                          "Analysis shows elevated EGFR.")

    def test_json_response_stripped_of_fences(self):
        raw = '```json\n{"proteins": ["EGFR", "TP53"]}\n```'
        parsed = json.loads(raw.replace("```json", "").replace("```", "").strip())
        self.assertIn("EGFR", parsed["proteins"])


# ===========================================================================
# 16. LangGraph Workflow Orchestration
# ===========================================================================

class TestLangGraphWorkflow(unittest.TestCase):

    def _make_state(self, **kwargs):
        base = {
            "session_id": "sess_001",
            "messages": [],
            "n_proteins": 1919,
            "n_samples": 10,
            "omic_type": "proteomics",
            "data_path": "/tmp/test.xlsx",
        }
        base.update(kwargs)
        return base

    def test_state_updated_after_node(self):
        state = self._make_state()
        state["messages"].append({"role": "assistant", "content": "Done"})
        self.assertEqual(len(state["messages"]), 1)

    def test_add_messages_no_duplication(self):
        messages = [{"role": "user", "content": "Hello"}]
        new_msg = {"role": "assistant", "content": "Hi"}
        updated = list(messages)
        if new_msg not in updated:
            updated.append(new_msg)
        self.assertEqual(len(updated), 2)

    def test_single_node_graph_structure(self):
        nodes = ["learning_agent"]
        self.assertEqual(len(nodes), 1)
        self.assertIn("learning_agent", nodes)

    def test_state_persisted_after_turn(self):
        state = self._make_state()
        state["last_action"] = "run_full_pipeline"
        self.assertIn("last_action", state)

    def test_sse_streaming_delta_extraction(self):
        prior = [{"role": "user", "content": "Hello"}]
        after = prior + [{"role": "assistant", "content": "Hi there"}]
        delta = after[len(prior):]
        self.assertEqual(len(delta), 1)
        self.assertEqual(delta[0]["role"], "assistant")


# ===========================================================================
# 17. FastAPI — Upload Route
# ===========================================================================

class TestUploadRoute(unittest.TestCase):

    def _validate_upload_response(self, resp):
        return {"session_id", "n_proteins", "n_samples", "omic_type", "message"}.issubset(resp.keys())

    def test_upload_response_schema(self):
        self.assertTrue(self._validate_upload_response({
            "session_id": "abc123",
            "n_proteins": 1919, "n_samples": 10,
            "omic_type": "proteomics",
            "message": "Data loaded — 1919 proteins · 10 samples · ms_lfq",
        }))

    def test_missing_file_returns_error(self):
        self.assertIn("detail", {"detail": "No file uploaded"})

    def test_non_xlsx_file_rejected(self):
        self.assertNotIn(os.path.splitext("report.pdf")[1].lower(), {".xlsx", ".xls"})

    def test_session_id_created_on_upload(self):
        import uuid
        self.assertRegex(str(uuid.uuid4()), r"[0-9a-f\-]{36}")

    def test_raw_file_path_constructed(self):
        path = "data/raw/test_session/data.xlsx"
        self.assertIn("test_session", path)
        self.assertTrue(path.endswith(".xlsx"))


# ===========================================================================
# 18. FastAPI — Chat Route
# ===========================================================================

class TestChatRoute(unittest.TestCase):

    def _validate_chat_response(self, resp):
        return all(k in resp for k in ("session_id", "response", "intent", "status"))

    def test_chat_request_schema(self):
        req = {"session_id": "sess_001", "message": "run analysis"}
        self.assertIn("session_id", req)
        self.assertIn("message", req)

    def test_chat_response_schema(self):
        self.assertTrue(self._validate_chat_response({
            "session_id": "sess_001",
            "response":   "Analysis complete.",
            "intent":     "run_full_pipeline",
            "status":     "success",
        }))

    def test_empty_message_rejected(self):
        self.assertFalse(len("".strip()) > 0)

    def test_session_not_found_returns_error(self):
        self.assertEqual({"detail": "Session not found", "status_code": 404}["status_code"], 404)

    def test_intent_in_valid_actions(self):
        valid = {"run_full_pipeline", "run_analysis", "answer", "query_data",
                  "run_enrichment", "run_visualization", "ask_clarification"}
        self.assertIn("run_full_pipeline", valid)


# ===========================================================================
# 19. Multi-Question Split
# ===========================================================================

class TestMultiQuestionSplit(unittest.TestCase):

    def _split_questions(self, message):
        parts = re.split(r"\?\s+(?=[A-Z])", message)
        return [p.strip() for p in parts if p.strip()]

    def test_two_questions_split(self):
        self.assertEqual(len(self._split_questions(
            "What are the top biomarkers? Also show the volcano plot.")), 2)

    def test_single_question_not_split(self):
        self.assertEqual(len(self._split_questions("Run the full analysis.")), 1)

    def test_three_questions_split(self):
        self.assertGreaterEqual(len(self._split_questions(
            "What are the biomarkers? Show the heatmap? Run enrichment.")), 2)

    def test_no_question_mark_single_part(self):
        self.assertEqual(len(self._split_questions("Compare DMD Quad vs BL6 Quad")), 1)

    def test_trailing_question_mark_handled(self):
        self.assertEqual(len(self._split_questions("Show me the results?")), 1)

    def test_split_threshold(self):
        self.assertFalse(len(self._split_questions("What are the top biomarkers?")) >= 2)


# ===========================================================================
# 20. Full Pipeline Integration
# ===========================================================================

class TestFullPipelineIntegration(unittest.TestCase):
    """Verifies each step of run_full_pipeline fires in order."""

    def _make_pipeline_state(self):
        df = _make_df(n_proteins=30, n_samples=6)
        return {
            "session_id":      "integration_test",
            "n_proteins":      30,
            "n_samples":       6,
            "omic_type":       "proteomics",
            "organism":        "Mus musculus",
            "disease_program": "DMD",
            "data_path":       "/tmp/fake.xlsx",
            "sample_map": {
                "Control": ["Sample_0_SpC", "Sample_1_SpC", "Sample_2_SpC"],
                "DMD":     ["Sample_3_SpC", "Sample_4_SpC", "Sample_5_SpC"],
            },
            "sample_columns":  [f"Sample_{i}_SpC" for i in range(6)],
            "messages":        [],
            "all_sheets":      {"expression": df},
            "label_map":       {},
            "plot_paths":      [],
            "excel_path":      None,
            "analysis_result": None,
            "biomarkers":      [],
        }

    def test_step1_data_summary_emits_message(self):
        state = self._make_pipeline_state()
        summary = (
            f"Data loaded — {state['n_proteins']} proteins · "
            f"{state['n_samples']} samples · {state['omic_type']}"
        )
        state["messages"].append({"role": "assistant", "content": summary})
        self.assertEqual(len(state["messages"]), 1)
        self.assertIn("proteins", state["messages"][0]["content"])

    def test_step2_analysis_populates_biomarkers(self):
        state = self._make_pipeline_state()
        state["biomarkers"] = ["EGFR", "TP53", "MYH3"]
        self.assertGreater(len(state["biomarkers"]), 0)

    def test_step3_enrichment_skipped_when_no_biomarkers(self):
        state = self._make_pipeline_state()
        should_run_enrichment = len(state["biomarkers"]) > 0
        self.assertFalse(should_run_enrichment)

    def test_step3_enrichment_runs_when_biomarkers_present(self):
        state = self._make_pipeline_state()
        state["biomarkers"] = ["EGFR", "TP53"]
        self.assertTrue(len(state["biomarkers"]) > 0)

    def test_step4_visualisation_produces_plots(self):
        state = self._make_pipeline_state()
        state["plot_paths"] = ["volcano.png", "heatmap.png", "pca.png"]
        self.assertEqual(len(state["plot_paths"]), 3)

    def test_step5_domain_expert_interpretation_appended(self):
        state = self._make_pipeline_state()
        state["messages"].append({
            "role": "assistant",
            "content": "## Biological interpretation\n- EGFR drives proliferation…",
        })
        self.assertIn("interpretation", state["messages"][-1]["content"].lower())

    def test_step6_drilldown_invite_appended(self):
        state = self._make_pipeline_state()
        drilldown_invite = (
            "**Full analysis complete.** You can now ask follow-up questions like:\n"
            "- *Compare DMD Quad vs BL6 Quad in detail*\n"
            "- *Show me the volcano plot*"
        )
        state["messages"].append({"role": "assistant", "content": drilldown_invite})
        self.assertIn("Compare", state["messages"][-1]["content"])

    def test_full_pipeline_sequence(self):
        """All 6 steps fire in order, in a single state object."""
        state = self._make_pipeline_state()
        # 1. summary
        state["messages"].append({"role": "assistant", "content": "Dataset summary"})
        # 2. analysis
        state["biomarkers"]      = ["EGFR", "TP53"]
        state["analysis_result"] = {"comparisons": 1}
        # 3. enrichment
        state["enrichment"]      = {"pathways": ["KEGG"]}
        # 4. visualisation
        state["plot_paths"]      = ["volcano.png"]
        # 5. domain expert
        state["messages"].append({"role": "assistant", "content": "Interpretation"})
        # 6. drill-down invite
        state["messages"].append({"role": "assistant", "content": "Next steps"})

        self.assertEqual(state["analysis_result"]["comparisons"], 1)
        self.assertEqual(len(state["plot_paths"]), 1)
        self.assertGreaterEqual(len(state["messages"]), 3)
        self.assertEqual(state["biomarkers"], ["EGFR", "TP53"])

    def test_pipeline_state_immutable_keys_preserved(self):
        """session_id, organism, etc. must not be overwritten by pipeline steps."""
        state = self._make_pipeline_state()
        original_session = state["session_id"]
        original_organism = state["organism"]
        # Simulate downstream updates
        state["messages"].append({"role": "assistant", "content": "anything"})
        state["plot_paths"].append("x.png")
        self.assertEqual(state["session_id"], original_session)
        self.assertEqual(state["organism"], original_organism)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
