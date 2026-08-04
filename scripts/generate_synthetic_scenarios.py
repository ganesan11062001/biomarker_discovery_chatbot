"""
scripts/generate_synthetic_scenarios.py

Generates one synthetic .xlsx per analysis scenario in the 10-case proteomics
evaluation rubric, in the same two-sheet template the app already parses
(matches tests/test_data/Solid Bio MSB-12244 041425.xlsx):

  Sheet 1 ("Proteins"):
    Row 0        : blank, blank, blank, <group label per sample column>...
    Row 1        : "Identified Proteins (N)", "Accession Number",
                   "Molecular Weight", <sample column name>...
    Row 2+       : protein description, accession, MW, <values>...

  Sheet 2 ("Sheet1"): blank, blank, then one unique group name per row.

Each file spikes a known subset of proteins with a real effect so the
generated log report can check "was the true signal actually recovered?",
not just "did it run without crashing".

Output: data/synthetic_scenarios/<NN>_<name>.xlsx
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

RNG = np.random.default_rng(2026)
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "synthetic_scenarios"


def _protein_names(n: int, prefix: str = "Protein") -> List[str]:
    return [
        f"{prefix}-{i} OS=Homo sapiens OX=9606 GN=GENE{i:04d} PE=1 SV=1"
        for i in range(1, n + 1)
    ]


def _accessions(n: int) -> List[str]:
    return [f"P{i:05d}" for i in range(1, n + 1)]


def _write_two_sheet_template(
    path: Path,
    group_per_sample: Sequence[str],
    sample_names: Sequence[str],
    protein_desc: Sequence[str],
    accession: Sequence[str],
    mol_weight: Sequence[str],
    values: np.ndarray,
) -> None:
    """values: shape (n_proteins, n_samples)."""
    assert len(group_per_sample) == len(sample_names) == values.shape[1]
    assert len(protein_desc) == len(accession) == len(mol_weight) == values.shape[0]

    n_id_cols = 3
    row0 = [None] * n_id_cols + list(group_per_sample)
    row1 = ["Identified Proteins (%d)" % values.shape[0], "Accession Number", "Molecular Weight"] + list(sample_names)
    data_rows = []
    for i in range(values.shape[0]):
        data_rows.append([protein_desc[i], accession[i], mol_weight[i]] + list(values[i]))

    sheet1_df = pd.DataFrame([row0, row1] + data_rows)

    unique_groups = list(dict.fromkeys(group_per_sample))
    legend_rows = [[None], [None]] + [[g] for g in unique_groups]
    legend_df = pd.DataFrame(legend_rows)

    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        sheet1_df.to_excel(writer, sheet_name="Proteins", header=False, index=False)
        legend_df.to_excel(writer, sheet_name="Sheet1", header=False, index=False)


def _sample_cols(groups: Dict[str, int], metric: str = "Intensity") -> tuple[List[str], List[str]]:
    """
    Return (group_per_sample, letter_coded_sample_names) for {group: n_reps}.

    `metric` becomes part of the column name and drives _detect_data_type's
    keyword match — use "NPX" for Olink-platform scenarios (log2-like scale,
    small values) so they aren't misclassified as ms_lfq just because the
    generic word "Intensity" would otherwise match first.
    """
    group_per_sample: List[str] = []
    sample_names: List[str] = []
    letters = iter("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    for gname, n in groups.items():
        letter = next(letters)
        for rep in range(1, n + 1):
            group_per_sample.append(gname)
            sample_names.append(f"{metric} {letter}{rep}")
    return group_per_sample, sample_names


# ── Scenario 1: single pooled plasma sample ────────────────────────────────
def scenario_1_pooled_plasma():
    n = 200
    groups = {"Pooled_Plasma": 1}
    gps, snames = _sample_cols(groups, metric="NPX")
    vals = RNG.normal(loc=5.0, scale=2.0, size=(n, 1))
    # A handful of unusually abundant proteins
    vals[:10] += RNG.uniform(4, 8, size=(10, 1))
    _write_two_sheet_template(
        OUT_DIR / "01_pooled_plasma_single_sample.xlsx",
        gps, snames, _protein_names(n, "OlinkProtein"), _accessions(n),
        ["-"] * n, vals,
    )


# ── Scenario 2: small case-control n=3v3 (Olink NPX) ───────────────────────
def scenario_2_case_control():
    n = 92
    groups = {"Healthy": 3, "Disease": 3}
    gps, snames = _sample_cols(groups, metric="NPX")
    vals = RNG.normal(loc=5.0, scale=1.0, size=(n, 6))
    spiked = list(range(8))
    vals[spiked, 3:] += RNG.uniform(2.5, 4.0, size=(len(spiked), 3))  # up in Disease
    _write_two_sheet_template(
        OUT_DIR / "02_case_control_n3v3.xlsx",
        gps, snames, _protein_names(n, "InflammationPanel"), _accessions(n),
        ["-"] * n, vals,
    )


# ── Scenario 3: paired pre/post treatment ──────────────────────────────────
def scenario_3_paired_pre_post():
    n = 92
    n_pairs = 6
    groups = {"Baseline": n_pairs, "PostTreatment": n_pairs}
    gps, snames = _sample_cols(groups, metric="NPX")
    baseline = RNG.normal(loc=5.0, scale=1.0, size=(n, n_pairs))
    post = baseline.copy()
    spiked = list(range(10))
    # Consistent within-subject decrease post-treatment (e.g. inflammation resolving)
    post[spiked] -= RNG.uniform(2.0, 3.5, size=(len(spiked), n_pairs))
    vals = np.concatenate([baseline, post], axis=1)
    _write_two_sheet_template(
        OUT_DIR / "03_paired_pre_post.xlsx",
        gps, snames, _protein_names(n, "InflammationPanel"), _accessions(n),
        ["-"] * n, vals,
    )


# ── Scenario 4: discovery MS n=5v5 ──────────────────────────────────────────
def scenario_4_discovery_ms():
    n = 300
    groups = {"Control": 5, "Disease": 5}
    gps, snames = _sample_cols(groups)
    log_vals = RNG.normal(loc=25.0, scale=1.5, size=(n, 10))  # log2 intensity-like
    spiked = list(range(15))
    log_vals[spiked, 5:] += RNG.uniform(2.0, 4.0, size=(len(spiked), 5))
    vals = np.power(2.0, log_vals)  # convert to raw MS-LFQ intensity scale
    _write_two_sheet_template(
        OUT_DIR / "04_discovery_ms_n5v5.xlsx",
        gps, snames, _protein_names(n, "TumorProtein"), _accessions(n),
        [f"{RNG.integers(10, 200)} kDa" for _ in range(n)], vals,
    )


# ── Scenario 5: TMT multiplex, 4 treatment groups ──────────────────────────
def scenario_5_tmt_multiplex():
    n = 250
    groups = {"Vehicle": 3, "TreatmentA": 3, "TreatmentB": 3, "TreatmentC": 3}
    gps, snames = _sample_cols(groups)
    log_vals = RNG.normal(loc=25.0, scale=1.2, size=(n, 12))
    # Omnibus ANOVA signal: differs across all 4 groups
    omnibus = list(range(10))
    log_vals[np.ix_(omnibus, range(3, 6))] += 2.5   # up in TreatmentA
    log_vals[np.ix_(omnibus, range(6, 9))] += 1.0   # mildly up in TreatmentB
    # Pairwise-specific signal: only TreatmentC differs from Vehicle (for Tukey)
    pairwise_only = list(range(10, 15))
    log_vals[np.ix_(pairwise_only, range(9, 12))] += 3.5
    vals = np.power(2.0, log_vals)
    _write_two_sheet_template(
        OUT_DIR / "05_tmt_multiplex_4groups.xlsx",
        gps, snames, _protein_names(n, "CellLineProtein"), _accessions(n),
        [f"{RNG.integers(10, 200)} kDa" for _ in range(n)], vals,
    )


# ── Scenario 6: dose-response, 4 dose levels ───────────────────────────────
def scenario_6_dose_response():
    n = 250
    groups = {"Dose0_Vehicle": 3, "Dose1_Low": 3, "Dose2_Medium": 3, "Dose3_High": 3}
    gps, snames = _sample_cols(groups)
    log_vals = RNG.normal(loc=25.0, scale=1.0, size=(n, 12))
    trend = list(range(12))
    dose_numeric = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    slope = RNG.uniform(0.8, 1.5, size=len(trend))
    for k, p in enumerate(trend):
        log_vals[p] += slope[k] * dose_numeric + RNG.normal(0, 0.2, size=12)
    vals = np.power(2.0, log_vals)
    _write_two_sheet_template(
        OUT_DIR / "06_dose_response_4levels.xlsx",
        gps, snames, _protein_names(n, "CellCultureProtein"), _accessions(n),
        [f"{RNG.integers(10, 200)} kDa" for _ in range(n)], vals,
    )


# ── Scenario 7: time-course, 5 timepoints, 4 matched subjects ──────────────
def scenario_7_time_course():
    n = 200
    days = ["Day0", "Day1", "Day3", "Day7", "Day14"]
    n_subj = 4
    groups: Dict[str, int] = {d: n_subj for d in days}
    gps, _ = _sample_cols(groups)
    # Descriptive sample names so subject correspondence is human-readable
    snames = [f"{d}_Subj{s}" for d in days for s in range(1, n_subj + 1)]
    log_vals = RNG.normal(loc=25.0, scale=1.0, size=(n, len(snames)))
    responders = list(range(10))
    day_index = np.array([days.index(s.split("_")[0]) for s in snames])
    # Peaks at Day3 then declines — realistic acute-response time-course shape
    peak_curve = np.array([0.0, 2.0, 3.0, 1.0, 0.2])
    for p in responders:
        log_vals[p] += peak_curve[day_index] + RNG.normal(0, 0.15, size=len(snames))
    vals = np.power(2.0, log_vals)
    _write_two_sheet_template(
        OUT_DIR / "07_time_course_5timepoints.xlsx",
        gps, snames, _protein_names(n, "PlasmaProtein"), _accessions(n),
        ["-"] * n, vals,
    )


# ── Scenario 8: disease-severity subtypes ──────────────────────────────────
def scenario_8_disease_subtypes():
    n = 92
    groups = {"Healthy": 4, "Mild": 4, "Moderate": 4, "Severe": 4}
    gps, snames = _sample_cols(groups, metric="NPX")
    vals = RNG.normal(loc=5.0, scale=1.0, size=(n, 16))
    graded = list(range(12))
    severity_step = np.array([0, 1, 2, 3])
    step_effect = RNG.uniform(0.8, 1.8, size=len(graded))
    for k, p in enumerate(graded):
        col_severity = np.repeat(severity_step, 4)
        vals[p] += step_effect[k] * col_severity + RNG.normal(0, 0.2, size=16)
    _write_two_sheet_template(
        OUT_DIR / "08_disease_severity_subtypes.xlsx",
        gps, snames, _protein_names(n, "SerumProtein"), _accessions(n),
        ["-"] * n, vals,
    )


# ── Scenario 9: phosphoproteomics (site-level identifiers) ─────────────────
def scenario_9_phosphoproteomics():
    n = 150
    groups = {"Tumor": 4, "Normal": 4}
    gps, snames = _sample_cols(groups)
    log_vals = RNG.normal(loc=22.0, scale=1.3, size=(n, 8))
    spiked = list(range(12))
    log_vals[spiked, 4:] += RNG.uniform(2.0, 3.5, size=(len(spiked), 4))
    vals = np.power(2.0, log_vals)

    kinases = ["EGFR", "AKT1", "MAPK1", "MTOR", "GSK3B", "STAT3", "SRC", "PDPK1"]
    residues = ["S", "T", "Y"]
    site_ids = [
        f"{kinases[i % len(kinases)]}_{residues[i % len(residues)]}{100 + i}"
        for i in range(n)
    ]
    protein_desc = [
        f"{site} phosphosite OS=Homo sapiens OX=9606 GN={site} PE=1 SV=1"
        for site in site_ids
    ]
    _write_two_sheet_template(
        OUT_DIR / "09_phosphoproteomics_ptm.xlsx",
        gps, snames, protein_desc, _accessions(n),
        ["-"] * n, vals,
    )


# ── Scenario 10: large clinical cohort (responder vs non-responder) ───────
def scenario_10_clinical_cohort():
    n = 150
    n_resp, n_nonresp = 60, 60
    groups = {"Responder": n_resp, "NonResponder": n_nonresp}
    gps, snames = _sample_cols(groups, metric="NPX")
    vals = RNG.normal(loc=5.0, scale=1.2, size=(n, n_resp + n_nonresp))
    predictive = list(range(20))
    vals[predictive, :n_resp] += RNG.uniform(1.0, 2.5, size=(len(predictive), n_resp))
    _write_two_sheet_template(
        OUT_DIR / "10_large_clinical_cohort.xlsx",
        gps, snames, _protein_names(n, "PlasmaBiomarker"), _accessions(n),
        ["-"] * n, vals,
    )


SCENARIOS = [
    ("1", "Single pooled plasma sample", scenario_1_pooled_plasma),
    ("2", "Small case-control n=3v3", scenario_2_case_control),
    ("3", "Paired pre/post treatment", scenario_3_paired_pre_post),
    ("4", "Discovery MS n=5v5", scenario_4_discovery_ms),
    ("5", "TMT multiplex 4 groups", scenario_5_tmt_multiplex),
    ("6", "Dose-response 4 levels", scenario_6_dose_response),
    ("7", "Time-course 5 timepoints", scenario_7_time_course),
    ("8", "Disease-severity subtypes", scenario_8_disease_subtypes),
    ("9", "Phosphoproteomics/PTM", scenario_9_phosphoproteomics),
    ("10", "Large clinical cohort", scenario_10_clinical_cohort),
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for num, name, fn in SCENARIOS:
        fn()
        print(f"[{num:>2}] {name} — generated")
    print(f"\nAll {len(SCENARIOS)} synthetic scenario files written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
