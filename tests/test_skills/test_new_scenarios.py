"""
tests/test_skills/test_new_scenarios.py

Unit tests for the 5 new statistical scenarios added to close gaps in the
10-case proteomics evaluation rubric:
  Case 5  — Tukey HSD post-hoc for one-way ANOVA
  Case 6  — dose-response linear trend test
  Case 7  — repeated-measures / time-course analysis (AnovaRM + mixed-effects fallback)
  Case 9  — PTM / kinase-substrate enrichment library selection
  Case 10 — clinical regression: linear / logistic (+ ROC AUC) / Cox

All tests use synthetic data and make no external network calls (gseapy's
Enrichr call is monkeypatched).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest

from skills.proteomics_analysis import ProteomicsAnalysisSkill


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture()
def anova_csv(tmp_path: Path):
    """3 groups (WT/KO/HET) x 3 replicates, 10 proteins.
    P001 is elevated only in KO (WT vs KO and HET vs KO should differ;
    WT vs HET should not)."""
    rng = np.random.default_rng(seed=7)
    cols = ["WT1", "WT2", "WT3", "KO1", "KO2", "KO3", "HET1", "HET2", "HET3"]
    proteins = [f"P{i:03d}" for i in range(1, 11)]
    data = rng.normal(loc=10.0, scale=0.4, size=(10, 9))
    data[0, 3:6] += 5.0  # P001 elevated only in KO
    df = pd.DataFrame(data, index=proteins, columns=cols)
    path = tmp_path / "anova.csv"
    df.to_csv(path)
    return path, cols


@pytest.fixture()
def dose_csv(tmp_path: Path):
    """4 dose groups (Vehicle/Low/Medium/High) x 3 replicates, 10 proteins.
    P001 increases monotonically with dose."""
    rng = np.random.default_rng(seed=11)
    cols = [f"{g}{i}" for g in ("Veh", "Low", "Med", "High") for i in (1, 2, 3)]
    proteins = [f"P{i:03d}" for i in range(1, 11)]
    data = rng.normal(loc=10.0, scale=0.3, size=(10, 12))
    dose_effect = np.repeat([0.0, 1.0, 2.0, 3.0], 3)
    data[0, :] += dose_effect * 2.0  # strong linear trend for P001
    df = pd.DataFrame(data, index=proteins, columns=cols)
    path = tmp_path / "dose.csv"
    df.to_csv(path)
    return path, cols


@pytest.fixture()
def timecourse_csv(tmp_path: Path):
    """3 time points (Day0/Day3/Day7) x 4 subjects tracked longitudinally.
    P001 increases over time within each subject."""
    rng = np.random.default_rng(seed=13)
    subjects = ["S1", "S2", "S3", "S4"]
    times = ["Day0", "Day3", "Day7"]
    cols = [f"{s}_{t}" for t in times for s in subjects]
    proteins = [f"P{i:03d}" for i in range(1, 11)]
    data = rng.normal(loc=10.0, scale=0.3, size=(10, 12))
    time_effect = {"Day0": 0.0, "Day3": 1.5, "Day7": 3.0}
    for j, col in enumerate(cols):
        t = col.split("_")[1]
        data[0, j] += time_effect[t]
    df = pd.DataFrame(data, index=proteins, columns=cols)
    path = tmp_path / "timecourse.csv"
    df.to_csv(path)
    all_groups = {t: [f"{s}_{t}" for s in subjects] for t in times}
    subject_map = {f"{s}_{t}": s for t in times for s in subjects}
    return path, cols, all_groups, subject_map


@pytest.fixture()
def clinical_csv(tmp_path: Path):
    """20 samples, 10 proteins. P001 correlates with a continuous score,
    separates responders/non-responders, and correlates with survival time."""
    rng = np.random.default_rng(seed=17)
    n = 20
    cols = [f"Pt{i}" for i in range(1, n + 1)]
    proteins = [f"P{i:03d}" for i in range(1, 11)]
    data = rng.normal(loc=10.0, scale=1.0, size=(10, n))
    signal = rng.normal(loc=0.0, scale=1.0, size=n)
    data[0, :] += signal * 2.0  # P001 carries the predictive signal
    df = pd.DataFrame(data, index=proteins, columns=cols)
    path = tmp_path / "clinical.csv"
    df.to_csv(path)

    continuous_outcome = {c: float(3.0 * s + rng.normal(scale=0.5)) for c, s in zip(cols, signal)}
    binary_outcome = {c: int(s > 0) for c, s in zip(cols, signal)}
    cox_outcome = {
        c: {"time": float(max(1.0, 20.0 - 3.0 * s + rng.normal(scale=1.0))),
            "event": int(rng.integers(0, 2))}
        for c, s in zip(cols, signal)
    }
    return path, cols, continuous_outcome, binary_outcome, cox_outcome


# ── Case 5: ANOVA + Tukey HSD post-hoc ────────────────────────────────────────

class TestTukeyPostHoc:
    def test_anova_includes_tukey_column(self, anova_csv, tmp_path):
        path, cols = anova_csv
        all_groups = {"WT": cols[0:3], "KO": cols[3:6], "HET": cols[6:9]}
        skill = ProteomicsAnalysisSkill()
        result = skill.execute(
            data_path=str(path),
            sample_columns=cols,
            group1_samples=[], group2_samples=[],
            analysis_mode="supervised",
            test_method="anova",
            all_groups=all_groups,
            output_dir=str(tmp_path),
        )
        assert result.get("error") is None
        assert len(result["top_biomarkers"]) > 0
        assert "tukey_significant_pairs" in result["top_biomarkers"][0]

    def test_tukey_flags_correct_pair_for_spiked_protein(self, anova_csv, tmp_path):
        path, cols = anova_csv
        all_groups = {"WT": cols[0:3], "KO": cols[3:6], "HET": cols[6:9]}
        skill = ProteomicsAnalysisSkill()
        result = skill.execute(
            data_path=str(path),
            sample_columns=cols,
            group1_samples=[], group2_samples=[],
            analysis_mode="supervised",
            test_method="anova",
            all_groups=all_groups,
            adj_pval_cutoff=0.05,
            log2fc_cutoff=0.5,
            output_dir=str(tmp_path),
        )
        p001 = next((b for b in result["top_biomarkers"] if b["protein"] == "P001"), None)
        assert p001 is not None, "P001 should be significant (elevated only in KO)"
        assert "KO" in p001["tukey_significant_pairs"]


# ── Case 6: dose-response trend test ───────────────────────────────────────────

class TestDoseResponse:
    def test_dose_response_returns_expected_columns(self, dose_csv, tmp_path):
        path, cols = dose_csv
        all_groups = {
            "Vehicle": cols[0:3], "Low": cols[3:6],
            "Medium": cols[6:9], "High": cols[9:12],
        }
        dose_levels = {"Vehicle": 0.0, "Low": 1.0, "Medium": 2.0, "High": 3.0}
        skill = ProteomicsAnalysisSkill()
        result = skill.execute(
            data_path=str(path),
            sample_columns=cols,
            group1_samples=[], group2_samples=[],
            analysis_mode="supervised",
            test_method="dose_response",
            all_groups=all_groups,
            dose_levels=dose_levels,
            output_dir=str(tmp_path),
        )
        assert result.get("error") is None
        bm = result["top_biomarkers"][0]
        for key in ("dose_slope", "r_squared", "trend_direction", "adj_p_value"):
            assert key in bm

    def test_spiked_protein_shows_increasing_trend(self, dose_csv, tmp_path):
        path, cols = dose_csv
        all_groups = {
            "Vehicle": cols[0:3], "Low": cols[3:6],
            "Medium": cols[6:9], "High": cols[9:12],
        }
        dose_levels = {"Vehicle": 0.0, "Low": 1.0, "Medium": 2.0, "High": 3.0}
        skill = ProteomicsAnalysisSkill()
        result = skill.execute(
            data_path=str(path),
            sample_columns=cols,
            group1_samples=[], group2_samples=[],
            analysis_mode="supervised",
            test_method="dose_response",
            all_groups=all_groups,
            dose_levels=dose_levels,
            output_dir=str(tmp_path),
        )
        p001 = next(b for b in result["top_biomarkers"] if b["protein"] == "P001")
        assert p001["trend_direction"] == "increasing"
        assert p001["dose_slope"] > 0
        assert p001["rank"] == 1  # should be the strongest hit

    def test_dose_response_requires_three_levels(self, dose_csv, tmp_path):
        path, cols = dose_csv
        all_groups = {"Vehicle": cols[0:3], "Low": cols[3:6]}
        dose_levels = {"Vehicle": 0.0, "Low": 1.0}
        skill = ProteomicsAnalysisSkill()
        result = skill.execute(
            data_path=str(path),
            sample_columns=cols,
            group1_samples=[], group2_samples=[],
            analysis_mode="supervised",
            test_method="dose_response",
            all_groups=all_groups,
            dose_levels=dose_levels,
            output_dir=str(tmp_path),
        )
        assert result.get("error") is not None


# ── Case 7: repeated-measures / time-course ────────────────────────────────────

class TestRepeatedMeasures:
    def test_repeated_measures_returns_expected_columns(self, timecourse_csv, tmp_path):
        path, cols, all_groups, subject_map = timecourse_csv
        skill = ProteomicsAnalysisSkill()
        result = skill.execute(
            data_path=str(path),
            sample_columns=cols,
            group1_samples=[], group2_samples=[],
            analysis_mode="supervised",
            test_method="repeated_measures",
            all_groups=all_groups,
            subject_map=subject_map,
            output_dir=str(tmp_path),
        )
        assert result.get("error") is None
        bm = result["top_biomarkers"][0]
        for key in ("f_statistic", "p_value", "adj_p_value", "method_used", "n_subjects"):
            assert key in bm

    def test_spiked_protein_significant_over_time(self, timecourse_csv, tmp_path):
        path, cols, all_groups, subject_map = timecourse_csv
        skill = ProteomicsAnalysisSkill()
        result = skill.execute(
            data_path=str(path),
            sample_columns=cols,
            group1_samples=[], group2_samples=[],
            analysis_mode="supervised",
            test_method="repeated_measures",
            all_groups=all_groups,
            subject_map=subject_map,
            adj_pval_cutoff=0.1,
            log2fc_cutoff=0.5,
            output_dir=str(tmp_path),
        )
        p001 = next((b for b in result["top_biomarkers"] if b["protein"] == "P001"), None)
        assert p001 is not None
        assert p001["adj_p_value"] < 0.1

    def test_unbalanced_design_falls_back_to_mixed_effects(self, timecourse_csv, tmp_path):
        path, cols, all_groups, subject_map = timecourse_csv
        # Drop one subject's Day7 sample to unbalance the design
        unbalanced_groups = dict(all_groups)
        unbalanced_groups["Day7"] = [c for c in all_groups["Day7"] if c != "S1_Day7"]
        skill = ProteomicsAnalysisSkill()
        result = skill.execute(
            data_path=str(path),
            sample_columns=cols,
            group1_samples=[], group2_samples=[],
            analysis_mode="supervised",
            test_method="repeated_measures",
            all_groups=unbalanced_groups,
            subject_map=subject_map,
            output_dir=str(tmp_path),
        )
        assert result.get("error") is None
        methods_used = {b["method_used"] for b in result["top_biomarkers"]}
        assert "mixed_effects_model" in methods_used


# ── Case 10: clinical regression ──────────────────────────────────────────────

class TestClinicalRegression:
    def test_linear_regression(self, clinical_csv, tmp_path):
        path, cols, continuous_outcome, _, _ = clinical_csv
        skill = ProteomicsAnalysisSkill()
        result = skill.execute(
            data_path=str(path),
            sample_columns=cols,
            group1_samples=[], group2_samples=[],
            analysis_mode="supervised",
            test_method="linear_regression",
            clinical_outcome=continuous_outcome,
            output_dir=str(tmp_path),
        )
        assert result.get("error") is None
        bm = result["top_biomarkers"][0]
        for key in ("beta_coefficient", "r_squared", "adj_p_value"):
            assert key in bm
        p001 = next(b for b in result["top_biomarkers"] if b["protein"] == "P001")
        assert p001["adj_p_value"] < 0.05

    def test_logistic_regression_with_auc(self, clinical_csv, tmp_path):
        path, cols, _, binary_outcome, _ = clinical_csv
        skill = ProteomicsAnalysisSkill()
        result = skill.execute(
            data_path=str(path),
            sample_columns=cols,
            group1_samples=[], group2_samples=[],
            analysis_mode="supervised",
            test_method="logistic_regression",
            clinical_outcome=binary_outcome,
            output_dir=str(tmp_path),
        )
        assert result.get("error") is None
        bm = result["top_biomarkers"][0]
        for key in ("odds_ratio", "auc", "adj_p_value"):
            assert key in bm
        p001 = next(b for b in result["top_biomarkers"] if b["protein"] == "P001")
        assert p001["auc"] > 0.7  # strong separating signal by construction

    def test_cox_regression(self, clinical_csv, tmp_path):
        path, cols, _, _, cox_outcome = clinical_csv
        skill = ProteomicsAnalysisSkill()
        result = skill.execute(
            data_path=str(path),
            sample_columns=cols,
            group1_samples=[], group2_samples=[],
            analysis_mode="supervised",
            test_method="cox_regression",
            clinical_outcome=cox_outcome,
            output_dir=str(tmp_path),
        )
        assert result.get("error") is None
        bm = result["top_biomarkers"][0]
        for key in ("hazard_ratio", "adj_p_value"):
            assert key in bm

    def test_requires_minimum_samples(self, clinical_csv, tmp_path):
        path, cols, continuous_outcome, _, _ = clinical_csv
        tiny_outcome = {c: continuous_outcome[c] for c in cols[:2]}
        skill = ProteomicsAnalysisSkill()
        result = skill.execute(
            data_path=str(path),
            sample_columns=cols,
            group1_samples=[], group2_samples=[],
            analysis_mode="supervised",
            test_method="linear_regression",
            clinical_outcome=tiny_outcome,
            output_dir=str(tmp_path),
        )
        assert result.get("error") is not None


# ── Case 9: PTM / kinase enrichment library selection ─────────────────────────

class TestPTMEnrichmentLibraries:
    def test_ptm_omic_type_adds_kinase_libraries(self, monkeypatch, tmp_path):
        from skills.run_enrichment import PathwaySkill, _PTM_LIBRARIES

        captured_libs: List[str] = []

        class _FakeEnrichrResult:
            def __init__(self):
                self.results = pd.DataFrame({
                    "Term": [], "Adjusted P-value": [], "Genes": [],
                })

        def _fake_enrichr(gene_list, gene_sets, organism, outdir, background, cutoff, verbose):
            captured_libs.append(gene_sets)
            return _FakeEnrichrResult()

        import gseapy
        monkeypatch.setattr(gseapy, "enrichr", _fake_enrichr)

        skill = PathwaySkill()
        skill.execute(
            protein_list=["sp|P12345|ABC1_HUMAN", "sp|P23456|XYZ2_HUMAN"],
            organism="human",
            output_dir=str(tmp_path),
            omic_type="phosphoproteomics",
        )
        for lib in _PTM_LIBRARIES:
            assert lib in captured_libs

    def test_non_ptm_omic_type_excludes_kinase_libraries(self, monkeypatch, tmp_path):
        from skills.run_enrichment import PathwaySkill, _PTM_LIBRARIES

        captured_libs: List[str] = []

        class _FakeEnrichrResult:
            def __init__(self):
                self.results = pd.DataFrame({
                    "Term": [], "Adjusted P-value": [], "Genes": [],
                })

        def _fake_enrichr(gene_list, gene_sets, organism, outdir, background, cutoff, verbose):
            captured_libs.append(gene_sets)
            return _FakeEnrichrResult()

        import gseapy
        monkeypatch.setattr(gseapy, "enrichr", _fake_enrichr)

        skill = PathwaySkill()
        skill.execute(
            protein_list=["sp|P12345|ABC1_HUMAN", "sp|P23456|XYZ2_HUMAN"],
            organism="human",
            output_dir=str(tmp_path),
            omic_type="proteomics",
        )
        for lib in _PTM_LIBRARIES:
            assert lib not in captured_libs
