"""
tests/test_agents/test_biomarker_agent.py

Unit tests for BiomarkerAgent — routing, state management,
and registry integration.  LLM calls are mocked.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from agents.biomarker_agent import BiomarkerAgent
from skills.omics_registry import OmicsSkillRegistry


def _make_state(overrides: dict | None = None) -> dict:
    base: dict = {
        "session_id": "test-session",
        "messages":   [],
        "status":     "routed",
        "data_path":  None,
        "omic_type":  "proteomics",
    }
    if overrides:
        base.update(overrides)
    return base


class TestBiomarkerAgentRegistry:
    def test_registry_contains_proteomics(self):
        agent = BiomarkerAgent()
        assert "proteomics" in agent._registry

    def test_registry_type(self):
        agent = BiomarkerAgent()
        assert isinstance(agent._registry, OmicsSkillRegistry)


class TestBiomarkerAgentRun:
    def test_no_data_path_returns_error(self):
        agent = BiomarkerAgent()
        state = _make_state()
        result = agent.run(state)
        assert result["status"] == "error"
        assert result["messages"][-1]["role"] == "assistant"

    def test_unsupported_omic_type_returns_error(self, tmp_path):
        agent = BiomarkerAgent()
        state = _make_state({
            "data_path": str(tmp_path / "fake.csv"),
            "omic_type": "metabolomics",
        })
        result = agent.run(state)
        assert result["status"] == "error"
        assert "metabolomics" in result["error_message"]

    def test_supervised_mode_detected(
        self, proteomics_csv, sample_columns, group1_samples, group2_samples
    ):
        agent = BiomarkerAgent()
        with patch.object(agent, "_call_llm", return_value="Mock summary."):
            state = _make_state({
                "data_path":      str(proteomics_csv),
                "sample_columns": sample_columns,
                "group1_samples": group1_samples,
                "group2_samples": group2_samples,
                "group1_label":   "Disease",
                "group2_label":   "Control",
                "omic_type":      "proteomics",
            })
            result = agent.run(state)

        assert result["status"] == "analysis_complete"
        assert result["analysis_mode"] == "supervised"
        assert isinstance(result["top_biomarkers"], list)

    def test_unsupervised_mode_when_no_groups(self, proteomics_csv, sample_columns):
        agent = BiomarkerAgent()
        with patch.object(agent, "_call_llm", return_value="Mock summary."):
            state = _make_state({
                "data_path":      str(proteomics_csv),
                "sample_columns": sample_columns,
                "omic_type":      "proteomics",
            })
            result = agent.run(state)

        assert result["analysis_mode"] == "unsupervised"
        assert result["status"] == "analysis_complete"

    def test_result_state_fields_populated(
        self, proteomics_csv, sample_columns, group1_samples, group2_samples
    ):
        agent = BiomarkerAgent()
        with patch.object(agent, "_call_llm", return_value="Summary."):
            state = _make_state({
                "data_path":      str(proteomics_csv),
                "sample_columns": sample_columns,
                "group1_samples": group1_samples,
                "group2_samples": group2_samples,
            })
            result = agent.run(state)

        for field in ("top_biomarkers", "n_significant", "excel_path",
                      "qc_summary", "analysis_summary", "omic_type"):
            assert field in result, f"Missing result field: {field}"
