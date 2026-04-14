"""
tests/test_skills/test_proteomics_analysis.py

Unit tests for ProteomicsAnalysisSkill — the core analysis pipeline.
These tests use only synthetic data and make no external calls.
"""
from __future__ import annotations

from pathlib import Path

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
