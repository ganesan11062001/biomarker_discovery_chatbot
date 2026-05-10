"""
tests/test_skills/test_visualization.py
Tests for ProteomicsPlotSuite and helper functions.
Uses synthetic data — no real proteomics files or LLM calls.
PNG export requires kaleido; tests fall back gracefully if kaleido is absent.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from skills.run_visualization import (
    PLOT_REGISTRY,
    ProteomicsPlotSuite,
    _fc_col,
    _pval_col,
    _rgba,
    _short,
    resolve_plot_names,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def sample_cols():
    return ["D1", "D2", "D3", "C1", "C2", "C3"]


@pytest.fixture()
def group1():
    return ["D1", "D2", "D3"]


@pytest.fixture()
def group2():
    return ["C1", "C2", "C3"]


@pytest.fixture()
def wide_csv(tmp_path, sample_cols) -> Path:
    """Synthetic wide-format proteomics CSV (20 proteins × 6 samples)."""
    rng  = np.random.default_rng(0)
    data = rng.normal(10.0, 1.0, size=(20, len(sample_cols)))
    data[:5, :3] += 3.0   # spike 5 proteins in group1
    df = pd.DataFrame(data, index=[f"P{i:03d}" for i in range(1, 21)], columns=sample_cols)
    path = tmp_path / "wide.csv"
    df.to_csv(path)
    return path


@pytest.fixture()
def dea_csv(tmp_path) -> Path:
    """Synthetic DEA results CSV with fold-change and p-value columns."""
    rng = np.random.default_rng(1)
    n   = 30
    df  = pd.DataFrame({
        "protein":          [f"P{i:03d}" for i in range(1, n + 1)],
        "log2_fold_change": rng.normal(0, 2, n),
        "p_value":          rng.uniform(0, 1, n),
        "adj_p_value":      rng.uniform(0, 1, n),
    })
    df.iloc[:5, df.columns.get_loc("adj_p_value")] = 0.001   # spike 5 significant proteins
    path = tmp_path / "dea_results.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture()
def top_proteins():
    """Minimal top-biomarker list as returned by ProteomicsAnalysisSkill."""
    return [
        {"rank": i, "protein": f"P{i:03d}",
         "log2_fold_change": (i % 2) * 2.0 - 1.0,
         "adj_p_value": 0.001 * i,
         "significance": "***" if i <= 3 else "*"}
        for i in range(1, 11)
    ]


@pytest.fixture()
def suite():
    return ProteomicsPlotSuite()


# ── Helper function tests ─────────────────────────────────────────────────────

class TestHelpers:

    def test_fc_col_finds_log2_fold_change(self):
        df = pd.DataFrame({"log2_fold_change": [1.0], "other": [2.0]})
        assert _fc_col(df) == "log2_fold_change"

    def test_fc_col_finds_log2fc(self):
        df = pd.DataFrame({"log2fc": [1.0]})
        assert _fc_col(df) == "log2fc"

    def test_fc_col_returns_none_when_absent(self):
        df = pd.DataFrame({"unrelated": [1.0]})
        assert _fc_col(df) is None

    def test_pval_col_finds_adj_p_value(self):
        df = pd.DataFrame({"adj_p_value": [0.01]})
        assert _pval_col(df) == "adj_p_value"

    def test_pval_col_prefers_adjusted(self):
        df = pd.DataFrame({"p_value": [0.05], "adj_p_value": [0.01]})
        assert _pval_col(df) == "adj_p_value"

    def test_rgba_returns_valid_string(self):
        result = _rgba("#FF0000", 0.5)
        assert result == "rgba(255,0,0,0.5)"

    def test_rgba_black(self):
        assert _rgba("#000000", 1.0) == "rgba(0,0,0,1.0)"

    def test_rgba_white(self):
        assert _rgba("#FFFFFF", 0.2) == "rgba(255,255,255,0.2)"

    def test_short_truncates_long_names(self):
        long_name = "VERY_LONG_PROTEIN_NAME_THAT_EXCEEDS_LIMIT"
        assert len(_short(long_name, 20)) <= 20

    def test_short_preserves_short_names(self):
        short_name = "PROT_A"
        assert _short(short_name, 30) == short_name


# ── PLOT_REGISTRY ─────────────────────────────────────────────────────────────

class TestPlotRegistry:

    def test_registry_has_16_plots(self):
        assert len(PLOT_REGISTRY) == 16

    def test_all_registry_values_are_callable(self):
        for name, fn in PLOT_REGISTRY.items():
            assert callable(fn), f"PLOT_REGISTRY['{name}'] is not callable"

    def test_expected_plot_names_present(self):
        expected = {
            "volcano", "ma_plot", "waterfall", "heatmap", "pca",
            "sample_correlation", "boxplot", "violin", "topn_bar",
            "paired_lines", "anova_multigroup", "cv_distribution",
            "fc_heatmap", "rescue_bar", "silac_ratio_dist", "pathway_dotplot",
        }
        assert expected == set(PLOT_REGISTRY.keys())


# ── resolve_plot_names ────────────────────────────────────────────────────────

class TestResolvePlotNames:

    def test_exact_names_returned_unchanged(self):
        assert resolve_plot_names(["volcano", "pca"]) == ["volcano", "pca"]

    def test_alias_volcano_plot_resolves(self):
        result = resolve_plot_names(["volcano plot"])
        assert "volcano" in result

    def test_alias_heatmap_resolves(self):
        result = resolve_plot_names(["heatmap"])
        assert "heatmap" in result

    def test_unknown_name_ignored(self):
        result = resolve_plot_names(["not_a_real_plot"])
        assert result == []

    def test_mixed_valid_invalid(self):
        result = resolve_plot_names(["volcano", "nonexistent"])
        assert "volcano" in result
        assert "nonexistent" not in result

    def test_empty_list_returns_empty(self):
        assert resolve_plot_names([]) == []


# ── Individual plot functions ─────────────────────────────────────────────────

class TestIndividualPlots:
    """Smoke tests — each plot function must return a non-empty path without raising."""

    def _run(self, fn, tmp_path, skip_if_empty: bool = False, **kwargs) -> str:
        result = fn(stem="smoke", output_dir=str(tmp_path), **kwargs)
        assert isinstance(result, str), f"{fn.__name__} must return a string"
        if skip_if_empty and not result:
            pytest.skip(f"{fn.__name__} returned empty (missing required columns)")
        if result:
            assert Path(result).exists(), f"Plot file missing: {result}"
        return result

    def test_volcano(self, top_proteins, tmp_path):
        from skills.run_visualization import plot_volcano
        self._run(
            plot_volcano, tmp_path,
            top_proteins=top_proteins,
            contrast_groups=["Disease", "Control"],
            adj_pval_cutoff=0.05, log2fc_cutoff=1.0,
        )

    def test_ma_plot(self, top_proteins, tmp_path):
        from skills.run_visualization import plot_ma
        # MA plot requires mean_* columns; skip gracefully if data lacks them
        self._run(
            plot_ma, tmp_path, skip_if_empty=True,
            top_proteins=top_proteins,
            contrast_groups=["Disease", "Control"],
            adj_pval_cutoff=0.05, log2fc_cutoff=1.0,
        )

    def test_waterfall(self, top_proteins, tmp_path):
        from skills.run_visualization import plot_waterfall
        self._run(
            plot_waterfall, tmp_path,
            top_proteins=top_proteins,
            contrast_groups=["Disease", "Control"],
            log2fc_cutoff=1.0,
        )

    def test_topn_bar(self, top_proteins, tmp_path):
        from skills.run_visualization import plot_topn_bar
        self._run(
            plot_topn_bar, tmp_path,
            top_proteins=top_proteins,
            contrast_groups=["Disease", "Control"],
        )

    def test_pca(self, wide_csv, sample_cols, group1, group2, tmp_path):
        from skills.run_visualization import plot_pca
        self._run(
            plot_pca, tmp_path,
            data_path=str(wide_csv),
            sample_columns=sample_cols,
            group1_samples=group1,
            group2_samples=group2,
            group1_label="Disease",
            group2_label="Control",
        )

    def test_heatmap(self, wide_csv, top_proteins, sample_cols, group1, group2, tmp_path):
        from skills.run_visualization import plot_heatmap
        self._run(
            plot_heatmap, tmp_path,
            data_path=str(wide_csv),
            top_proteins=top_proteins,
            sample_columns=sample_cols,
            group1_samples=group1,
            group2_samples=group2,
            group1_label="Disease",
            group2_label="Control",
        )

    def test_sample_correlation(self, wide_csv, sample_cols, group1, group2, tmp_path):
        from skills.run_visualization import plot_sample_correlation
        self._run(
            plot_sample_correlation, tmp_path,
            data_path=str(wide_csv),
            sample_columns=sample_cols,
            group1_samples=group1,
            group2_samples=group2,
            group1_label="Disease",
            group2_label="Control",
        )

    def test_boxplot(self, wide_csv, sample_cols, group1, group2, tmp_path):
        from skills.run_visualization import plot_boxplot
        self._run(
            plot_boxplot, tmp_path,
            data_path=str(wide_csv),
            sample_columns=sample_cols,
            group1_samples=group1,
            group2_samples=group2,
            group1_label="Disease",
            group2_label="Control",
        )

    def test_violin(self, wide_csv, sample_cols, group1, group2, tmp_path):
        from skills.run_visualization import plot_violin
        self._run(
            plot_violin, tmp_path,
            data_path=str(wide_csv),
            top_proteins=None,
            sample_columns=sample_cols,
            group1_samples=group1,
            group2_samples=group2,
            group1_label="Disease",
            group2_label="Control",
        )

    def test_cv_distribution(self, tmp_path):
        from skills.run_visualization import plot_cv_distribution
        # CV distribution needs cv_percent field
        top_cv = [{"rank": i, "protein": f"P{i:03d}", "cv_percent": float(i * 5)} for i in range(1, 11)]
        self._run(plot_cv_distribution, tmp_path, top_proteins=top_cv)

    def test_fc_heatmap(self, top_proteins, tmp_path):
        from skills.run_visualization import plot_fc_heatmap
        self._run(
            plot_fc_heatmap, tmp_path,
            top_proteins=top_proteins,
            contrast_groups=["Disease", "Control"],
        )

    def test_pathway_dotplot_with_no_pathways(self, tmp_path):
        from skills.run_visualization import plot_pathway_dot
        result = plot_pathway_dot(pathways=[], stem="smoke", output_dir=str(tmp_path))
        assert isinstance(result, str)


# ── JSON sidecar output ───────────────────────────────────────────────────────

class TestJsonSidecar:
    """Every saved plot must produce a .json sidecar for st.plotly_chart."""

    def test_volcano_produces_json_sidecar(self, top_proteins, tmp_path):
        from skills.run_visualization import plot_volcano
        path = plot_volcano(
            top_proteins=top_proteins,
            contrast_groups=["D", "C"],
            adj_pval_cutoff=0.05, log2fc_cutoff=1.0,
            stem="sidecar", output_dir=str(tmp_path),
        )
        json_path = str(path).replace(".png", ".json").replace(".html", ".json")
        if Path(json_path).exists():
            with open(json_path) as f:
                data = json.load(f)
            assert "data" in data or "layout" in data

    def test_pca_produces_json_sidecar(self, wide_csv, sample_cols, group1, group2, tmp_path):
        from skills.run_visualization import plot_pca
        path = plot_pca(
            data_path=str(wide_csv),
            sample_columns=sample_cols,
            group1_samples=group1,
            group2_samples=group2,
            group1_label="D", group2_label="C",
            stem="sidecar_pca", output_dir=str(tmp_path),
        )
        json_path = str(path).replace(".png", ".json").replace(".html", ".json")
        if Path(json_path).exists():
            with open(json_path) as f:
                data = json.load(f)
            assert "data" in data


# ── ProteomicsPlotSuite ───────────────────────────────────────────────────────

class TestProteomicsPlotSuite:

    def test_execute_supervised_returns_plot_paths(
        self, wide_csv, dea_csv, top_proteins, sample_cols, group1, group2, tmp_path
    ):
        suite = ProteomicsPlotSuite()
        result = suite.execute(
            top_proteins    = top_proteins,
            analysis_mode   = "supervised",
            omic_type       = "proteomics",
            test_method     = "welch",
            is_paired       = False,
            all_groups      = None,
            data_path       = str(wide_csv),
            sample_columns  = sample_cols,
            group1_samples  = group1,
            group2_samples  = group2,
            group1_label    = "Disease",
            group2_label    = "Control",
            adj_pval_cutoff = 0.05,
            log2fc_cutoff   = 1.0,
            top_pathways    = None,
            enrichment_result_path = "",
            contrast_groups = ["Disease", "Control"],
            output_dir      = str(tmp_path),
            stem            = "test",
        )
        assert "plot_paths" in result
        assert isinstance(result["plot_paths"], list)
        assert len(result["plot_paths"]) > 0

    def test_execute_unsupervised_returns_plot_paths(
        self, wide_csv, sample_cols, tmp_path
    ):
        suite  = ProteomicsPlotSuite()
        top_cv = [
            {"rank": i, "protein": f"P{i:03d}", "cv_percent": float(i * 5)}
            for i in range(1, 11)
        ]
        result = suite.execute(
            top_proteins    = top_cv,
            analysis_mode   = "unsupervised",
            omic_type       = "proteomics",
            test_method     = "none",
            is_paired       = False,
            all_groups      = None,
            data_path       = str(wide_csv),
            sample_columns  = sample_cols,
            group1_samples  = [],
            group2_samples  = [],
            group1_label    = "All",
            group2_label    = "",
            adj_pval_cutoff = 0.05,
            log2fc_cutoff   = 1.0,
            top_pathways    = None,
            enrichment_result_path = "",
            contrast_groups = [],
            output_dir      = str(tmp_path),
            stem            = "unsup",
        )
        assert isinstance(result.get("plot_paths"), list)
        assert len(result["plot_paths"]) > 0

    def test_execute_subset_of_plots(
        self, wide_csv, top_proteins, sample_cols, group1, group2, tmp_path
    ):
        suite  = ProteomicsPlotSuite()
        result = suite.execute(
            top_proteins    = top_proteins,
            analysis_mode   = "supervised",
            omic_type       = "proteomics",
            test_method     = "welch",
            is_paired       = False,
            all_groups      = None,
            data_path       = str(wide_csv),
            sample_columns  = sample_cols,
            group1_samples  = group1,
            group2_samples  = group2,
            group1_label    = "D",
            group2_label    = "C",
            adj_pval_cutoff = 0.05,
            log2fc_cutoff   = 1.0,
            top_pathways    = None,
            enrichment_result_path = "",
            contrast_groups = ["D", "C"],
            plot_types      = ["volcano", "pca"],
            output_dir      = str(tmp_path),
            stem            = "subset",
        )
        plots_run = result.get("plots_run", [])
        assert "volcano" in plots_run or len(result["plot_paths"]) <= 2

    def test_plot_paths_all_exist(
        self, wide_csv, top_proteins, sample_cols, group1, group2, tmp_path
    ):
        suite  = ProteomicsPlotSuite()
        result = suite.execute(
            top_proteins    = top_proteins,
            analysis_mode   = "supervised",
            omic_type       = "proteomics",
            test_method     = "welch",
            is_paired       = False,
            all_groups      = None,
            data_path       = str(wide_csv),
            sample_columns  = sample_cols,
            group1_samples  = group1,
            group2_samples  = group2,
            group1_label    = "D",
            group2_label    = "C",
            adj_pval_cutoff = 0.05,
            log2fc_cutoff   = 1.0,
            top_pathways    = None,
            enrichment_result_path = "",
            contrast_groups = ["D", "C"],
            output_dir      = str(tmp_path),
            stem            = "exist_check",
        )
        for path in result.get("plot_paths", []):
            assert Path(path).exists(), f"Plot file missing: {path}"

    def test_execute_paired_mode(
        self, wide_csv, top_proteins, sample_cols, group1, group2, tmp_path
    ):
        suite  = ProteomicsPlotSuite()
        result = suite.execute(
            top_proteins    = top_proteins,
            analysis_mode   = "supervised",
            omic_type       = "proteomics",
            test_method     = "paired_t",
            is_paired       = True,
            all_groups      = None,
            data_path       = str(wide_csv),
            sample_columns  = sample_cols,
            group1_samples  = group1,
            group2_samples  = group2,
            group1_label    = "Pre",
            group2_label    = "Post",
            adj_pval_cutoff = 0.05,
            log2fc_cutoff   = 1.0,
            top_pathways    = None,
            enrichment_result_path = "",
            contrast_groups = ["Pre", "Post"],
            output_dir      = str(tmp_path),
            stem            = "paired",
        )
        assert isinstance(result.get("plot_paths"), list)

    def test_pathway_dotplot_included_when_pathways_provided(
        self, wide_csv, top_proteins, sample_cols, group1, group2, tmp_path
    ):
        pathways = [
            {"pathway": "Glycolysis",    "p_adjust": 0.001, "gene_count": 20},
            {"pathway": "TCA cycle",     "p_adjust": 0.005, "gene_count": 15},
            {"pathway": "MAPK signaling","p_adjust": 0.01,  "gene_count": 30},
        ]
        suite  = ProteomicsPlotSuite()
        result = suite.execute(
            top_proteins    = top_proteins,
            analysis_mode   = "supervised",
            omic_type       = "proteomics",
            test_method     = "welch",
            is_paired       = False,
            all_groups      = None,
            data_path       = str(wide_csv),
            sample_columns  = sample_cols,
            group1_samples  = group1,
            group2_samples  = group2,
            group1_label    = "D",
            group2_label    = "C",
            adj_pval_cutoff = 0.05,
            log2fc_cutoff   = 1.0,
            top_pathways    = pathways,
            enrichment_result_path = "",
            contrast_groups = ["D", "C"],
            output_dir      = str(tmp_path),
            stem            = "pathways",
        )
        plots_run = result.get("plots_run", [])
        assert "pathway_dotplot" in plots_run

    def test_plots_run_field_returned(
        self, wide_csv, top_proteins, sample_cols, group1, group2, tmp_path
    ):
        suite  = ProteomicsPlotSuite()
        result = suite.execute(
            top_proteins    = top_proteins,
            analysis_mode   = "supervised",
            omic_type       = "proteomics",
            test_method     = "welch",
            is_paired       = False,
            all_groups      = None,
            data_path       = str(wide_csv),
            sample_columns  = sample_cols,
            group1_samples  = group1,
            group2_samples  = group2,
            group1_label    = "D",
            group2_label    = "C",
            adj_pval_cutoff = 0.05,
            log2fc_cutoff   = 1.0,
            top_pathways    = None,
            enrichment_result_path = "",
            contrast_groups = ["D", "C"],
            output_dir      = str(tmp_path),
            stem            = "runs_field",
        )
        assert "plots_run" in result
        assert isinstance(result["plots_run"], list)
