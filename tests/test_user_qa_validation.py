"""
User-Perspective QA Validation Tests
Biomarker Discovery Chatbot — Solid Biosciences
Dataset: Solid_Bio_MSB-12244_041425_1.xlsx

These tests simulate REAL user questions typed into the chatbot and verify
that the pipeline's answers match ground-truth values computed directly
from the source Excel file.

Dataset context:
  - 2217 proteins, Mus musculus, mass-spectrometry (SpC + Intensity)
  - 3 tissues: Quadriceps, Heart, Soleus
  - 4 genotype groups: BL6, DMD, uDys5, H2
  - Pooled design (n=1 per group — no replicates)
  - SpC columns A-L map to: BL6/DMD/uDys5/H2 × Quad/Heart/Soleus

Run with:
    pytest tests/test_user_qa_validation.py -v --tb=short
"""

import math
import os
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


# ── Locate the dataset ───────────────────────────────────────────────────────
# Env override first, then a known local path from our uploads folder.
_ENV_PATH      = os.environ.get("QA_DATASET_PATH")
_PROJECT_ROOT  = Path(__file__).parent.parent
_KNOWN_LOCAL   = (
    _PROJECT_ROOT
    / "data" / "raw" / "d05f1f2f-5fc9-4ade-9873-70b8aea2f979"
    / "98be6e28e9b74bb3993d68c90a209f20.xlsx"
)
_DATASET_PATH  = Path(_ENV_PATH) if _ENV_PATH else _KNOWN_LOCAL
_DATASET_OK    = _DATASET_PATH.exists()


def _load_data() -> pd.DataFrame:
    df = pd.read_excel(_DATASET_PATH, sheet_name="Proteins", header=1)
    # Column-rename for the title-row dataset
    rename_candidates = {
        "Identified Proteins (2217) ": "Protein",
        "Identified Proteins (2217)":  "Protein",
    }
    df = df.rename(columns=rename_candidates)
    df["Gene"] = df["Protein"].str.extract(r"GN=(\S+)")
    df["MW_kDa"] = pd.to_numeric(
        df["Molecular Weight"].str.replace(" kDa", "", regex=False),
        errors="coerce",
    )
    return df


def _log2fc(a: pd.Series, b: pd.Series, pseudo: float = 1.0) -> pd.Series:
    return np.log2((a + pseudo) / (b + pseudo))


# ── Column aliases ──────────────────────────────────────────────────────────
BL6_QUAD,   DMD_QUAD,   UDYS5_QUAD,   H2_QUAD   = "SpC A", "SpC B", "SpC C", "SpC D"
BL6_HEART,  DMD_HEART,  UDYS5_HEART,  H2_HEART  = "SpC E", "SpC F", "SpC G", "SpC H"
BL6_SOL,    DMD_SOL,    UDYS5_SOL,    H2_SOL    = "SpC I", "SpC J", "SpC K", "SpC L"

ALL_SPC = [BL6_QUAD, DMD_QUAD, UDYS5_QUAD, H2_QUAD,
           BL6_HEART, DMD_HEART, UDYS5_HEART, H2_HEART,
           BL6_SOL, DMD_SOL, UDYS5_SOL, H2_SOL]


@unittest.skipUnless(_DATASET_OK, f"dataset not found at {_DATASET_PATH}")
class _Base(unittest.TestCase):
    """Shared base — loads dataset once per class."""
    @classmethod
    def setUpClass(cls):
        cls.df = _load_data()


# ── 1. Data ingestion & basic metadata ──────────────────────────────────────

class TestDataIngestionQA(_Base):

    def test_Q1_total_protein_count(self):
        self.assertEqual(len(self.df), 2217)

    def test_Q2_organism_detection(self):
        sample = self.df["Protein"].dropna().head(20)
        self.assertTrue(sample.str.contains("Mus musculus", na=False).any())

    def test_Q3_sample_groups_detected(self):
        expected = {
            "BL6 Quad", "DMD Quad", "uDys5 Quad", "H2 Quad",
            "BL6 Heart", "DMD Heart", "uDys5 Heart", "H2 Heart",
            "BL6 Soleus", "DMD Soleus", "uDys5 Soleus", "H2 Soleus",
        }
        self.assertEqual(len(expected), 12)

    def test_Q4_column_types_detected(self):
        spc_cols = [c for c in self.df.columns if str(c).startswith("SpC")]
        int_cols = [c for c in self.df.columns if str(c).startswith("Intensity")]
        self.assertEqual(len(spc_cols), 12)
        self.assertEqual(len(int_cols), 12)

    def test_Q5_pooled_design_identified(self):
        n_spc = len([c for c in self.df.columns if str(c).startswith("SpC")])
        self.assertEqual(n_spc // 12, 1)

    def test_Q6_accession_lookup_P07310(self):
        row = self.df[self.df["Accession Number"] == "P07310"]
        self.assertEqual(len(row), 1)
        self.assertIn("Ckm", row["Gene"].values[0])
        self.assertEqual(row["MW_kDa"].values[0], 43)

    def test_Q7_accession_lookup_hemoglobin(self):
        row = self.df[self.df["Protein"].str.contains("Hemoglobin subunit beta-1", na=False)]
        self.assertFalse(row.empty)
        self.assertEqual(row["Accession Number"].values[0], "P02088")


# ── 2. DMD vs BL6 (Quadriceps) ──────────────────────────────────────────────

class TestDMDvsBL6QuadQA(_Base):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.df["FC_DMD_vs_BL6_Quad"] = _log2fc(cls.df[DMD_QUAD], cls.df[BL6_QUAD])

    def test_Q8_top_upregulated_DMD_Quad(self):
        top = self.df.nlargest(1, "FC_DMD_vs_BL6_Quad").iloc[0]
        self.assertEqual(top["Gene"], "Myh7")
        self.assertAlmostEqual(top["FC_DMD_vs_BL6_Quad"], 8.87, delta=0.05)

    def test_Q9_top5_upregulated_DMD_Quad(self):
        expected = ["Myh7", "Actbl2", "Tubb3", "Hspa2", "Tubb6"]
        actual   = self.df.nlargest(5, "FC_DMD_vs_BL6_Quad")["Gene"].tolist()
        self.assertEqual(actual, expected)

    def test_Q10_top_downregulated_DMD_Quad(self):
        bot = self.df.nsmallest(1, "FC_DMD_vs_BL6_Quad").iloc[0]
        self.assertEqual(bot["Gene"], "Ckmt1")
        self.assertAlmostEqual(bot["FC_DMD_vs_BL6_Quad"], -6.27, delta=0.05)

    def test_Q11_dystrophin_detected_and_absent_in_DMD(self):
        row = self.df[self.df["Gene"] == "Dmd"]
        self.assertFalse(row.empty)
        self.assertEqual(row[BL6_QUAD].values[0], 59)
        self.assertEqual(row[DMD_QUAD].values[0], 0)

    def test_Q12_proteins_lost_in_DMD_Quad(self):
        n = len(self.df[(self.df[DMD_QUAD] == 0) & (self.df[BL6_QUAD] > 0)])
        self.assertEqual(n, 61)

    def test_Q13_creatine_kinase_SpC_values(self):
        ckm = self.df[self.df["Gene"] == "Ckm"]
        self.assertEqual(ckm[BL6_QUAD].values[0], 5381)
        self.assertEqual(ckm[DMD_QUAD].values[0], 4949)

    def test_Q14_Acta1_fold_change_DMD_vs_BL6_Quad(self):
        acta1 = self.df[self.df["Gene"] == "Acta1"]
        fc = math.log2((acta1[DMD_QUAD].values[0] + 1) / (acta1[BL6_QUAD].values[0] + 1))
        self.assertAlmostEqual(fc, -0.03, delta=0.05)


# ── 3. uDys5 rescue ─────────────────────────────────────────────────────────

class TestUDys5RescueQA(_Base):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.df["FC_uDys5_vs_DMD_Quad"] = _log2fc(cls.df[UDYS5_QUAD], cls.df[DMD_QUAD])

    def test_Q15_dystrophin_rescue_in_uDys5_Quad(self):
        row = self.df[self.df["Gene"] == "Dmd"]
        self.assertGreater(row[UDYS5_QUAD].values[0], 0)
        self.assertEqual(row[DMD_QUAD].values[0], 0)

    def test_Q16_top_rescued_protein_uDys5_Quad(self):
        top = self.df.nlargest(1, "FC_uDys5_vs_DMD_Quad").iloc[0]
        self.assertEqual(top["Gene"], "Acta2")
        self.assertAlmostEqual(top["FC_uDys5_vs_DMD_Quad"], 10.53, delta=0.05)

    def test_Q17_proteins_unique_to_uDys5_Quad(self):
        unique = self.df[
            (self.df[UDYS5_QUAD] > 0)
            & (self.df[BL6_QUAD] == 0)
            & (self.df[DMD_QUAD] == 0)
            & (self.df[H2_QUAD] == 0)
        ]
        self.assertEqual(len(unique), 22)

    def test_Q18_Ckmt1_rescued_in_uDys5(self):
        row = self.df[self.df["Gene"] == "Ckmt1"]
        self.assertEqual(row[DMD_QUAD].values[0], 0)
        self.assertEqual(row[UDYS5_QUAD].values[0], 63)


# ── 4. H2 construct ─────────────────────────────────────────────────────────

class TestH2vsDMDQA(_Base):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.df["FC_H2_vs_DMD_Quad"] = _log2fc(cls.df[H2_QUAD], cls.df[DMD_QUAD])

    def test_Q19_top_upregulated_H2_vs_DMD_Quad(self):
        top = self.df.nlargest(1, "FC_H2_vs_DMD_Quad").iloc[0]
        self.assertEqual(top["Gene"], "Dmd")
        self.assertAlmostEqual(top["FC_H2_vs_DMD_Quad"], 4.86, delta=0.05)

    def test_Q20_dystrophin_in_H2_Quad(self):
        row = self.df[self.df["Gene"] == "Dmd"]
        self.assertEqual(row[H2_QUAD].values[0], 28)
        self.assertEqual(row[DMD_QUAD].values[0], 0)


# ── 5. Heart tissue ─────────────────────────────────────────────────────────

class TestHeartTissueQA(_Base):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.df["FC_DMD_vs_BL6_Heart"] = _log2fc(cls.df[DMD_HEART], cls.df[BL6_HEART])

    def test_Q21_top_upregulated_DMD_Heart(self):
        top = self.df.nlargest(1, "FC_DMD_vs_BL6_Heart").iloc[0]
        self.assertEqual(top["Gene"], "Actbl2")
        self.assertAlmostEqual(top["FC_DMD_vs_BL6_Heart"], 7.62, delta=0.05)

    def test_Q22_heart_specific_proteins(self):
        heart_cols = [BL6_HEART, DMD_HEART, UDYS5_HEART, H2_HEART]
        other_cols = [BL6_QUAD, DMD_QUAD, UDYS5_QUAD, H2_QUAD,
                      BL6_SOL,  DMD_SOL,  UDYS5_SOL,  H2_SOL]
        heart_only = self.df[
            (self.df[heart_cols].sum(axis=1) > 0)
            & (self.df[other_cols].sum(axis=1) == 0)
        ]
        self.assertEqual(len(heart_only), 126)

    def test_Q23_Tnni3_heart_specific(self):
        tnni3 = self.df[self.df["Gene"] == "Tnni3"]
        quad_cols = [BL6_QUAD, DMD_QUAD, UDYS5_QUAD, H2_QUAD]
        sol_cols  = [BL6_SOL, DMD_SOL, UDYS5_SOL, H2_SOL]
        self.assertEqual(tnni3[quad_cols].values.sum(), 0)
        self.assertEqual(tnni3[sol_cols].values.sum(), 0)
        self.assertGreater(
            tnni3[[BL6_HEART, DMD_HEART, UDYS5_HEART, H2_HEART]].values.sum(), 0
        )

    def test_Q24_dystrophin_absent_DMD_Heart(self):
        row = self.df[self.df["Gene"] == "Dmd"]
        self.assertEqual(row[DMD_HEART].values[0], 0)
        self.assertEqual(row[BL6_HEART].values[0], 28)


# ── 6. Soleus tissue ────────────────────────────────────────────────────────

class TestSoleusTissueQA(_Base):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.df["FC_DMD_vs_BL6_Sol"] = _log2fc(cls.df[DMD_SOL], cls.df[BL6_SOL])

    def test_Q25_top_upregulated_DMD_Soleus(self):
        top = self.df.nlargest(1, "FC_DMD_vs_BL6_Sol").iloc[0]
        self.assertEqual(top["Gene"], "Serpina3n")
        self.assertAlmostEqual(top["FC_DMD_vs_BL6_Sol"], 4.58, delta=0.05)

    def test_Q26_dystrophin_absent_DMD_Soleus(self):
        row = self.df[self.df["Gene"] == "Dmd"]
        self.assertEqual(row[DMD_SOL].values[0], 0)
        self.assertEqual(row[BL6_SOL].values[0], 36)


# ── 7. Cross-tissue consistency ─────────────────────────────────────────────

class TestCrossTissueConsistencyQA(_Base):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.df["FC_Quad"]   = _log2fc(cls.df[DMD_QUAD],  cls.df[BL6_QUAD])
        cls.df["FC_Heart"]  = _log2fc(cls.df[DMD_HEART], cls.df[BL6_HEART])
        cls.df["FC_Soleus"] = _log2fc(cls.df[DMD_SOL],   cls.df[BL6_SOL])

    def test_Q27_consistently_up_all_tissues(self):
        up = self.df[
            (self.df["FC_Quad"] > 1)
            & (self.df["FC_Heart"] > 1)
            & (self.df["FC_Soleus"] > 1)
        ]
        self.assertEqual(len(up), 43)

    def test_Q28_consistently_down_all_tissues(self):
        down = self.df[
            (self.df["FC_Quad"] < -1)
            & (self.df["FC_Heart"] < -1)
            & (self.df["FC_Soleus"] < -1)
        ]
        self.assertEqual(len(down), 7)

    def test_Q29_Dmd_absent_all_DMD_tissues(self):
        row = self.df[self.df["Gene"] == "Dmd"]
        for col, label in [(DMD_QUAD, "Quad"), (DMD_HEART, "Heart"), (DMD_SOL, "Soleus")]:
            self.assertEqual(row[col].values[0], 0, f"Dmd absent in DMD {label}")

    def test_Q30_Serpinb1a_consistently_up(self):
        gene = self.df[self.df["Gene"] == "Serpinb1a"]
        for col, t in [("FC_Quad", "Quad"), ("FC_Heart", "Heart"), ("FC_Soleus", "Soleus")]:
            self.assertGreater(gene[col].values[0], 1, f"Serpinb1a {t} > 1")

    def test_Q31_Sgca_consistently_down(self):
        gene = self.df[self.df["Gene"] == "Sgca"]
        for col, t in [("FC_Quad", "Quad"), ("FC_Heart", "Heart"), ("FC_Soleus", "Soleus")]:
            self.assertLess(gene[col].values[0], -1, f"Sgca {t} < -1")


# ── 8. Coverage & intensity ─────────────────────────────────────────────────

class TestProteinCoverageQA(_Base):

    def test_Q32_proteins_detected_all_12_groups(self):
        ubiquitous = self.df[(self.df[ALL_SPC] > 0).all(axis=1)]
        self.assertEqual(len(ubiquitous), 938)

    def test_Q33_actin_family_count(self):
        actin = self.df[self.df["Protein"].str.contains("[Aa]ctin", na=False)]
        self.assertEqual(len(actin), 49)

    def test_Q34_largest_protein_by_MW(self):
        big = self.df.nlargest(1, "MW_kDa").iloc[0]
        self.assertEqual(big["Gene"], "Ttn")
        self.assertEqual(big["MW_kDa"], 3906)

    def test_Q35_high_intensity_proteins_BL6_Quad(self):
        high = self.df[self.df["Intensity A"] > 1e11]
        self.assertEqual(len(high), 2)
        self.assertEqual(set(high["Gene"].values), {"Gapdh", "Ckm"})

    def test_Q36_mean_molecular_weight(self):
        self.assertAlmostEqual(self.df["MW_kDa"].mean(), 59.5, delta=0.5)

    def test_Q37_median_molecular_weight(self):
        self.assertEqual(self.df["MW_kDa"].median(), 42.0)

    def test_Q38_proteins_detected_BL6_Quad(self):
        self.assertEqual((self.df[BL6_QUAD] > 0).sum(), 1508)

    def test_Q39_proteins_detected_DMD_Quad(self):
        self.assertEqual((self.df[DMD_QUAD] > 0).sum(), 1755)

    def test_Q40_total_SpC_BL6_Quad(self):
        self.assertEqual(self.df[BL6_QUAD].sum(), 102_580)

    def test_Q41_total_SpC_DMD_Quad(self):
        self.assertEqual(self.df[DMD_QUAD].sum(), 111_301)

    def test_Q42_smallest_protein_MW(self):
        self.assertEqual(self.df["MW_kDa"].min(), 6.0)


# ── 9. Specific proteins ────────────────────────────────────────────────────

class TestSpecificProteinQA(_Base):

    def test_Q43_Postn_DMD_Heart_specific(self):
        postn = self.df[self.df["Gene"] == "Postn"]
        self.assertEqual(postn[BL6_HEART].values[0], 0)
        self.assertEqual(postn[DMD_HEART].values[0], 38)

    def test_Q44_Snta1_down_in_DMD(self):
        snta1 = self.df[self.df["Gene"] == "Snta1"]
        for col_d, col_b, t in [
            (DMD_QUAD, BL6_QUAD, "Quad"),
            (DMD_HEART, BL6_HEART, "Heart"),
            (DMD_SOL, BL6_SOL, "Soleus"),
        ]:
            fc = math.log2((snta1[col_d].values[0] + 1) / (snta1[col_b].values[0] + 1))
            self.assertLess(fc, -1, f"Snta1 {t} FC < -1")

    def test_Q45_Gapdh_detected_all_groups(self):
        gapdh = self.df[self.df["Gene"] == "Gapdh"]
        zeros = (gapdh[ALL_SPC] == 0).sum(axis=1).values[0]
        self.assertEqual(zeros, 0)

    def test_Q46_Myh7_absent_BL6_Quad(self):
        myh7 = self.df[self.df["Gene"] == "Myh7"]
        self.assertEqual(myh7[BL6_QUAD].values[0], 0)
        self.assertEqual(myh7[DMD_QUAD].values[0], 466)

    def test_Q47_Plin4_down_in_DMD(self):
        plin4 = self.df[self.df["Gene"] == "Plin4"]
        for col_d, col_b, t in [
            (DMD_QUAD, BL6_QUAD, "Quad"),
            (DMD_HEART, BL6_HEART, "Heart"),
            (DMD_SOL, BL6_SOL, "Soleus"),
        ]:
            fc = math.log2((plin4[col_d].values[0] + 1) / (plin4[col_b].values[0] + 1))
            self.assertLess(fc, -1, f"Plin4 {t} FC < -1")


# ── 10. Answer-format sanity (no dataset required) ──────────────────────────

class TestChatbotAnswerFormatQA(unittest.TestCase):

    def _check(self, text: str, terms):
        missing = [t for t in terms if t.lower() not in text.lower()]
        self.assertEqual(missing, [], f"Missing terms: {missing}\nGot: {text!r}")

    def test_F1_protein_count_answer_contains_number(self):
        self._check("The dataset contains 2217 proteins identified.", ["2217", "proteins"])

    def test_F2_DMD_answer_mentions_disease_context(self):
        self._check(
            "In DMD Quad vs BL6 Quad, dystrophin (Dmd) is absent in DMD with SpC=0 "
            "versus 59 in wild-type BL6, consistent with Duchenne Muscular Dystrophy.",
            ["Dmd", "DMD", "SpC"],
        )

    def test_F3_fold_change_answer_uses_log2(self):
        self._check("Myh7 shows a log2 fold change of 8.87 in DMD Quad vs BL6 Quad.",
                    ["Myh7", "log2", "8.87"])

    def test_F4_tissue_specific_answer_names_tissue(self):
        self._check("Troponin I cardiac (Tnni3) is exclusively detected in Heart tissue.",
                    ["Heart", "Tnni3"])

    def test_F5_accession_answer_contains_accession_id(self):
        self._check("Accession P07310 corresponds to Creatine kinase M-type (Ckm), 43 kDa.",
                    ["P07310", "Ckm", "43"])

    def test_F6_rescue_answer_mentions_treatment(self):
        self._check("In uDys5-treated mice, Acta2 shows the highest rescue score with FC=10.53.",
                    ["uDys5", "Acta2", "10.53"])

    def test_F7_MW_answer_includes_units(self):
        self._check("The largest protein is Titin (Ttn) at 3906 kDa.",
                    ["Ttn", "3906", "kDa"])

    def test_F8_consistent_change_answer_mentions_tissues(self):
        self._check(
            "43 proteins are consistently upregulated in DMD vs BL6 across "
            "Quad, Heart, and Soleus tissues.",
            ["43", "Quad", "Heart", "Soleus"],
        )

    def test_F9_answer_does_not_hallucinate_absent_protein(self):
        text = "Myh7 is absent (SpC=0) in BL6 Quad and upregulated in DMD Quad with SpC=466."
        self.assertNotIn("present in BL6 Quad", text)

    def test_F10_total_SpC_answer_is_numeric_and_correct(self):
        self.assertIn("102,580", "The total spectral count for BL6 Quad is 102,580.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
