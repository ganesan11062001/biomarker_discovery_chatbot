"""
tests/test_dynamic_detection.py
Tests for the runtime metadata detectors in core.proteomics_tools:
  - detect_organism()
  - detect_software()
  - detect_disease()

Every test uses synthetic data built inside the test — no real files
or hardcoded production answers.
"""
from __future__ import annotations

import pandas as pd
import pytest

from core.proteomics_tools import (
    detect_disease,
    detect_organism,
    detect_software,
)


# ── detect_organism ──────────────────────────────────────────────────────────

class TestDetectOrganism:

    def test_returns_species_from_os_field(self):
        sheets = {
            "Proteins": pd.DataFrame({
                "Protein Name": [
                    "P1 OS=Mus musculus OX=10090 GN=A PE=1 SV=1",
                    "P2 OS=Mus musculus GN=B PE=1 SV=1",
                ],
            }),
        }
        assert detect_organism(sheets) == "Mus musculus"

    def test_picks_most_common_species(self):
        sheets = {
            "Proteins": pd.DataFrame({
                "Protein description": [
                    "P1 OS=Rattus norvegicus GN=A",
                    "P2 OS=Mus musculus GN=B",
                    "P3 OS=Mus musculus GN=C",
                    "P4 OS=Mus musculus GN=D",
                ],
            }),
        }
        assert detect_organism(sheets) == "Mus musculus"

    def test_returns_none_when_no_os_field(self):
        sheets = {
            "Proteins": pd.DataFrame({
                "Some col": ["a", "b", "c"],
            }),
        }
        assert detect_organism(sheets) is None

    def test_empty_input(self):
        assert detect_organism({}) is None


# ── detect_software ──────────────────────────────────────────────────────────

class TestDetectSoftware:

    def test_maxquant_signature(self):
        sheets = {
            "Identifier Info": pd.DataFrame({
                "MaxQuant": ["A", "B"],
                "Client identifier": ["x", "y"],
            }),
        }
        assert detect_software(sheets) == "MaxQuant"

    def test_lfq_intensity_signature(self):
        # 'LFQ Intensity' columns also indicate MaxQuant
        sheets = {
            "Proteins": pd.DataFrame({
                "Protein IDs":    ["P1"],
                "LFQ Intensity A": [1.0],
                "LFQ Intensity B": [2.0],
            }),
        }
        assert detect_software(sheets) == "MaxQuant"

    def test_fragpipe_signature(self):
        sheets = {
            "FragPipe Report": pd.DataFrame({
                "Protein": ["P1"],
                "Intensity": [1.0],
            }),
        }
        assert detect_software(sheets) == "FragPipe"

    def test_spectronaut_signature(self):
        sheets = {
            "Proteins": pd.DataFrame({
                "PG.ProteinAccessions": ["P1"],
                "PG.Quantity":          [1.0],
            }),
        }
        assert detect_software(sheets) == "Spectronaut"

    def test_diann_signature(self):
        sheets = {
            "Report": pd.DataFrame({
                "Protein.Names":          ["P1"],
                "Precursor.Normalised":   [1.0],
            }),
        }
        assert detect_software(sheets) == "DIA-NN"

    def test_returns_none_for_unknown(self):
        sheets = {
            "Sheet1": pd.DataFrame({"foo": [1, 2, 3]}),
        }
        assert detect_software(sheets) is None

    def test_returns_software_with_most_hits(self):
        # MaxQuant has multiple signature hits, FragPipe has one — MaxQuant wins
        sheets = {
            "Identifier Info": pd.DataFrame({
                "MaxQuant":   ["A"],
                "LFQ Intensity": [1.0],
                "iBAQ":       [1.0],
                # FragPipe partial match
                "FragPipe Notes": ["x"],
            }),
        }
        result = detect_software(sheets)
        assert result == "MaxQuant"


# ── detect_disease ───────────────────────────────────────────────────────────

class TestDetectDisease:

    def test_mdx_in_strain_maps_to_dmd(self):
        sample_map = {
            "A": {"strain": "BL10 WT", "treatment": "Vehicle"},
            "B": {"strain": "MDX",     "treatment": "Vehicle"},
        }
        assert detect_disease(sample_map=sample_map) == "Duchenne Muscular Dystrophy (DMD)"

    def test_dystrophin_token_in_treatment(self):
        sample_map = {
            "A": {"client_id": "WT",  "treatment": "Vehicle"},
            "B": {"client_id": "KO",  "treatment": "AAV-dystrophin"},
        }
        assert detect_disease(sample_map=sample_map) == "Duchenne Muscular Dystrophy (DMD)"

    def test_sod1_maps_to_als(self):
        sample_map = {
            "A": {"strain": "WT"},
            "B": {"strain": "SOD1 G93A"},
        }
        assert detect_disease(sample_map=sample_map) == "Amyotrophic Lateral Sclerosis (ALS)"

    def test_smn1_maps_to_sma(self):
        sample_map = {
            "A": {"strain": "WT"},
            "B": {"client_id": "SMN1-deficient"},
        }
        assert detect_disease(sample_map=sample_map) == "Spinal Muscular Atrophy (SMA)"

    def test_htt_maps_to_huntingtons(self):
        sample_map = {
            "A": {"strain": "WT"},
            "B": {"strain": "HTT Q175"},
        }
        assert detect_disease(sample_map=sample_map) == "Huntington's Disease (HD)"

    def test_returns_none_when_no_signal(self):
        sample_map = {
            "A": {"strain": "WT", "treatment": "Vehicle"},
            "B": {"strain": "WT", "treatment": "Drug"},
        }
        assert detect_disease(sample_map=sample_map) is None

    def test_handles_empty_sample_map(self):
        assert detect_disease(sample_map={}) is None
        assert detect_disease() is None

    def test_falls_back_to_scanning_sheets(self):
        # No sample_map — should scan the identifier sheet directly
        identifier = pd.DataFrame({
            "Strain": ["BL10 WT", "MDX", "MDX"],
            "Group":  ["Vehicle",  "Vehicle", "Vehicle"],
        })
        sheets = {"Identifier Info": identifier}
        assert detect_disease(sample_map=None, all_sheets=sheets) == \
            "Duchenne Muscular Dystrophy (DMD)"

    def test_whole_word_matching(self):
        # 'dmd' inside 'dmdomain' must NOT match (whole-word boundary)
        sample_map = {
            "A": {"strain": "WT", "treatment": "dmdomain analysis"},
        }
        # No real signal — should return None
        assert detect_disease(sample_map=sample_map) is None

    def test_strongest_match_wins_when_multiple_diseases(self):
        # If both DMD and ALS tokens appear, the one with more hits wins
        sample_map = {
            "A": {"strain": "MDX", "treatment": "mdx vehicle", "client_id": "dystrophin study"},
            "B": {"strain": "SOD1"},
        }
        # DMD has 3 hits (mdx, mdx, dystrophin), ALS has 1 (sod1)
        assert detect_disease(sample_map=sample_map) == "Duchenne Muscular Dystrophy (DMD)"
