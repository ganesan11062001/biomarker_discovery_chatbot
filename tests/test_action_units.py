"""
tests/test_action_units.py
Tests for the action-units loader.
"""
from __future__ import annotations

import pytest

from core.action_units import ActionUnit, ActionUnitSet, load_action_units


class TestLoadActionUnits:

    def test_biomarker_units_load(self):
        u = load_action_units("biomarker")
        assert len(u.units) >= 5
        names = u.names()
        assert "qc_filter" in names
        assert "differential_expression" in names
        assert "fdr_correction" in names

    def test_ingestion_units_load(self):
        u = load_action_units("ingestion")
        assert "load_workbook" in u.names()
        assert "infer_groups" in u.names()

    def test_query_data_units_load(self):
        u = load_action_units("query_data")
        names = u.names()
        assert "generate_pandas" in names
        assert "review_code" in names
        assert "revise_on_failure" in names

    def test_missing_set_returns_empty(self):
        u = load_action_units("nonexistent_agent")
        assert u.units == []

    def test_by_name_finds_unit(self):
        u = load_action_units("biomarker").by_name("qc_filter")
        assert u is not None
        assert "missing" in u.instruction.lower()

    def test_as_prompt_block_renders(self):
        block = load_action_units("biomarker").as_prompt_block(
            only_names=["qc_filter", "fdr_correction"],
        )
        assert "qc_filter" in block
        assert "fdr_correction" in block
        # Other units should not appear
        assert "median_normalize" not in block
