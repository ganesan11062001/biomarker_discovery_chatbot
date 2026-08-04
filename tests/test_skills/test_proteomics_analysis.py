"""
tests/test_skills/test_proteomics_analysis.py

Unit tests for ProteomicsAnalysisSkill — the core analysis pipeline.
These tests use only synthetic data and make no external calls.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from skills.base_skill import BaseOmicsSkill, OmicsAnalysisResult
from skills.omics_registry import OmicsSkillRegistry
from skills.proteomics_analysis import ProteomicsAnalysisSkill


# ── Contract tests ────────────────────────────────────────────────────────────

class TestBaseOmicsSkillContract:
    """Verify ProteomicsAnalysisSkill honours the BaseOmicsSkill contract."""

    def test_is_base_omic_skill(self):
        assert issubclass(ProteomicsAnalysisSkill, BaseOmicsSkill)

    def test_omic_type_is_proteomics(self):
        skill = ProteomicsAnalysisSkill()
        assert skill.omic_type == "proteomics"

    def test_repr(self):
        skill = ProteomicsAnalysisSkill()
        assert "proteomics" in repr(skill)


# ── Registry tests ────────────────────────────────────────────────────────────

class TestOmicsSkillRegistry:
    def test_register_and_get(self):
        registry = OmicsSkillRegistry()
        registry.register(ProteomicsAnalysisSkill())
        skill = registry.get("proteomics")
        assert isinstance(skill, ProteomicsAnalysisSkill)

    def test_available_lists_registered_types(self):
        registry = OmicsSkillRegistry()
        registry.register(ProteomicsAnalysisSkill())
        assert "proteomics" in registry.available()

    def test_get_unknown_raises_key_error(self):
        registry = OmicsSkillRegistry()
        with pytest.raises(KeyError, match="transcriptomics"):
            registry.get("transcriptomics")

    def test_get_or_default_falls_back(self):
        registry = OmicsSkillRegistry()
        registry.register(ProteomicsAnalysisSkill())
        skill = registry.get_or_default("transcriptomics", default="proteomics")
        assert skill.omic_type == "proteomics"

    def test_contains(self):
        registry = OmicsSkillRegistry()
        registry.register(ProteomicsAnalysisSkill())
        assert "proteomics" in registry
        assert "metabolomics" not in registry

    def test_register_wrong_type_raises(self):
        registry = OmicsSkillRegistry()
        with pytest.raises(TypeError):
            registry.register("not_a_skill")  # type: ignore


# ── Supervised analysis ───────────────────────────────────────────────────────

class TestProteomicsAnalysisSkillSupervised:
    def test_supervised_returns_top_biomarkers(
        self, proteomics_csv, sample_columns, group1_samples, group2_samples, tmp_path
    ):
        skill = ProteomicsAnalysisSkill()
        result = skill.execute(
            data_path=str(proteomics_csv),
            sample_columns=sample_columns,
            group1_samples=group1_samples,
            group2_samples=group2_samples,
            group1_label="Disease",
            group2_label="Control",
            analysis_mode="supervised",
            output_dir=str(tmp_path),
        )
        assert result.get("error") is None
        assert result["omic_type"] == "proteomics"
        assert isinstance(result["top_biomarkers"], list)
        assert len(result["top_biomarkers"]) > 0
        assert result["n_significant"] >= 0

    def test_supervised_excel_file_created(
        self, proteomics_csv, sample_columns, group1_samples, group2_samples, tmp_path
    ):
        skill = ProteomicsAnalysisSkill()
        result = skill.execute(
            data_path=str(proteomics_csv),
            sample_columns=sample_columns,
            group1_samples=group1_samples,
            group2_samples=group2_samples,
            output_dir=str(tmp_path),
        )
        assert result.get("excel_path") is not None
        assert Path(result["excel_path"]).exists()

    def test_supervised_biomarker_fields(
        self, proteomics_csv, sample_columns, group1_samples, group2_samples, tmp_path
    ):
        skill = ProteomicsAnalysisSkill()
        result = skill.execute(
            data_path=str(proteomics_csv),
            sample_columns=sample_columns,
            group1_samples=group1_samples,
            group2_samples=group2_samples,
            output_dir=str(tmp_path),
        )
        required_keys = {"protein", "rank", "log2_fold_change", "adj_p_value", "significance"}
        for bm in result["top_biomarkers"]:
            assert required_keys.issubset(bm.keys()), f"Missing keys in {bm}"

    def test_spiked_proteins_rank_top(
        self, proteomics_csv, sample_columns, group1_samples, group2_samples, tmp_path
    ):
        """P001–P003 were artificially elevated; they should rank near the top."""
        skill = ProteomicsAnalysisSkill()
        result = skill.execute(
            data_path=str(proteomics_csv),
            sample_columns=sample_columns,
            group1_samples=group1_samples,
            group2_samples=group2_samples,
            output_dir=str(tmp_path),
        )
        top3_proteins = {b["protein"] for b in result["top_biomarkers"][:5]}
        assert top3_proteins & {"P001", "P002", "P003"}, (
            f"Expected spiked proteins in top 5, got {top3_proteins}"
        )


# ── Imputation must never fabricate significance ──────────────────────────────

class TestImputationSignificanceGuard:
    """
    A protein detected in every sample of one group but in none of the other
    is imputed to a constant (half-minimum) in the all-missing group. That
    constant has ~zero within-group variance, which can drive a t-test to an
    artificially tiny p-value that reflects the imputation, not real biology.
    Such rows must be flagged "Insufficient Data", never "Significant".
    """

    @pytest.fixture()
    def zero_detection_csv(self, tmp_path: Path) -> Path:
        rng = np.random.default_rng(seed=7)
        proteins = [f"P{i:03d}" for i in range(1, 11)]
        cols = ["D1", "D2", "D3", "C1", "C2", "C3"]
        data = rng.normal(loc=10.0, scale=1.0, size=(10, 6))
        df = pd.DataFrame(data, index=proteins, columns=cols)
        # P001: fully detected in D1-D3, completely missing in C1-C3 —
        # half-min imputation will fill C1-C3 with an identical constant.
        df.loc["P001", ["C1", "C2", "C3"]] = np.nan
        path = tmp_path / "zero_detection.csv"
        df.to_csv(path)
        return path

    def test_zero_detection_group_never_significant(self, zero_detection_csv, tmp_path):
        skill = ProteomicsAnalysisSkill()
        result = skill.execute(
            data_path=str(zero_detection_csv),
            sample_columns=["D1", "D2", "D3", "C1", "C2", "C3"],
            group1_samples=["D1", "D2", "D3"],
            group2_samples=["C1", "C2", "C3"],
            group1_label="Disease",
            group2_label="Control",
            analysis_mode="supervised",
            output_dir=str(tmp_path),
        )
        assert result.get("error") is None
        p001 = next(b for b in result["top_biomarkers"] if b["protein"] == "P001")
        assert p001["significance"] == "Insufficient Data"
        assert p001["adj_p_value"] == 1.0

    def test_anova_zero_detection_group_never_significant(self, tmp_path):
        rng = np.random.default_rng(seed=11)
        proteins = [f"P{i:03d}" for i in range(1, 11)]
        cols = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"]
        data = rng.normal(loc=10.0, scale=1.0, size=(10, 9))
        df = pd.DataFrame(data, index=proteins, columns=cols)
        df.loc["P001", ["C1", "C2", "C3"]] = np.nan
        path = tmp_path / "anova_zero_detection.csv"
        df.to_csv(path)

        skill = ProteomicsAnalysisSkill()
        result = skill.execute(
            data_path=str(path),
            sample_columns=cols,
            analysis_mode="supervised",
            test_method="anova",
            all_groups={"A": ["A1", "A2", "A3"], "B": ["B1", "B2", "B3"], "C": ["C1", "C2", "C3"]},
            output_dir=str(tmp_path),
        )
        assert result.get("error") is None
        p001 = next((b for b in result["top_biomarkers"] if b["protein"] == "P001"), None)
        if p001 is not None:
            assert p001["significance"] == "Insufficient Data"


# ── Unsupervised analysis ─────────────────────────────────────────────────────

class TestProteomicsAnalysisSkillUnsupervised:
    def test_unsupervised_returns_results(
        self, proteomics_csv, sample_columns, tmp_path
    ):
        skill = ProteomicsAnalysisSkill()
        result = skill.execute(
            data_path=str(proteomics_csv),
            sample_columns=sample_columns,
            analysis_mode="unsupervised",
            output_dir=str(tmp_path),
        )
        assert result.get("error") is None
        assert len(result["top_biomarkers"]) > 0

    def test_unsupervised_biomarker_has_cv_field(
        self, proteomics_csv, sample_columns, tmp_path
    ):
        skill = ProteomicsAnalysisSkill()
        result = skill.execute(
            data_path=str(proteomics_csv),
            sample_columns=sample_columns,
            analysis_mode="unsupervised",
            output_dir=str(tmp_path),
        )
        for bm in result["top_biomarkers"]:
            assert "cv_percent" in bm


# ── QC summary ────────────────────────────────────────────────────────────────

class TestQCSummary:
    def test_qc_summary_keys_present(
        self, proteomics_csv, sample_columns, group1_samples, group2_samples, tmp_path
    ):
        skill = ProteomicsAnalysisSkill()
        result = skill.execute(
            data_path=str(proteomics_csv),
            sample_columns=sample_columns,
            group1_samples=group1_samples,
            group2_samples=group2_samples,
            output_dir=str(tmp_path),
        )
        qc = result["qc_summary"]
        for key in ("proteins_input", "proteins_after_qc", "log2_transformed"):
            assert key in qc, f"Missing QC key: {key}"


# ── Error handling ────────────────────────────────────────────────────────────

class TestErrorHandling:
    def test_missing_file_returns_error(self, tmp_path):
        skill = ProteomicsAnalysisSkill()
        result = skill.execute(
            data_path="/nonexistent/path.csv",
            sample_columns=["S1", "S2"],
            output_dir=str(tmp_path),
        )
        assert result.get("error") is not None
        assert result["top_biomarkers"] == []

    def test_empty_sample_columns_returns_error(self, proteomics_csv, tmp_path):
        skill = ProteomicsAnalysisSkill()
        result = skill.execute(
            data_path=str(proteomics_csv),
            sample_columns=["NON_EXISTENT_COL"],
            output_dir=str(tmp_path),
        )
        assert result.get("error") is not None
