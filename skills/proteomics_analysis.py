"""
skills/proteomics_analysis.py
Analysis Layer – ProteomicsAnalysisSkill

Pure-Python proteomics biomarker analysis (no R required).
  1. QC  : missing-value filter, log2 transform, half-minimum imputation
  2. Stats: Welch t-test (supervised) or CV ranking (unsupervised)
  3. FDR : Benjamini-Hochberg correction
  4. Excel: formatted multi-sheet workbook output

Inherits BaseOmicsSkill — satisfies the multi-omic registry contract.
To add a new omic type, create a sibling class that also inherits
BaseOmicsSkill and register it in agents/biomarker_agent.py.
"""
from __future__ import annotations

import logging
import traceback
from datetime import datetime

logger = logging.getLogger(__name__)
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from skills.base_skill import BaseOmicsSkill, OmicsAnalysisResult


# ── Colour palette ────────────────────────────────────────────────────────────
_HEADER_FG = "FFFFFF"
_HEADER_BG = "1F4E79"
_HIGHLY_SIG = "00B050"   # dark green
_SIGNIFICANT = "92D050"  # light green
_TREND       = "FFEB9C"  # pale yellow
_VARIABLE    = "BDD7EE"  # pale blue (unsupervised top proteins)


def _thin_border() -> Border:
    s = Side(style="thin")
    return Border(left=s, right=s, top=s, bottom=s)


def _header_style() -> Tuple[PatternFill, Font]:
    fill = PatternFill(start_color=_HEADER_BG, end_color=_HEADER_BG, fill_type="solid")
    font = Font(bold=True, color=_HEADER_FG, size=11)
    return fill, font


class ProteomicsAnalysisSkill(BaseOmicsSkill):
    """
    Pure-Python proteomics biomarker discovery skill.

    Implements the BaseOmicsSkill interface so it can be registered in
    OmicsSkillRegistry and dispatched by BiomarkerAgent alongside future
    omic-type skills (transcriptomics, metabolomics, lipidomics, …).

    Usage
    -----
    skill = ProteomicsAnalysisSkill()
    result = skill.execute(
        data_path="data/processed/myfile_processed.csv",
        sample_columns=["S1","S2","S3","S4","S5","S6"],
        group1_samples=["S1","S2","S3"],
        group2_samples=["S4","S5","S6"],
        group1_label="Disease",
        group2_label="Control",
    )
    # result["excel_path"]     → absolute path to formatted Excel file
    # result["top_biomarkers"] → ranked list of biomarker dicts
    """

    @property
    def omic_type(self) -> str:
        return "proteomics"

    def execute(  # type: ignore[override]  # kwargs-based, matches BaseOmicsSkill
        self,
        data_path: str,
        sample_columns: List[str],
        group1_samples: Optional[List[str]] = None,
        group2_samples: Optional[List[str]] = None,
        group1_label: str = "Group1",
        group2_label: str = "Group2",
        analysis_mode: str = "supervised",
        data_type: str = "generic",
        adj_pval_cutoff: float = 0.05,
        log2fc_cutoff: float = 1.0,
        missing_threshold: float = 0.5,
        top_n: int = 50,
        output_dir: str = "outputs",
        file_name: str = "analysis",
    ) -> Dict[str, Any]:
        try:
            result = self._run(
                data_path, sample_columns, group1_samples, group2_samples,
                group1_label, group2_label, analysis_mode, data_type,
                adj_pval_cutoff, log2fc_cutoff, missing_threshold,
                top_n, output_dir, file_name,
            )
            result["omic_type"] = self.omic_type
            return result
        except Exception as exc:
            # Log full traceback for debugging; return only the clean message to the user.
            logger.error("ProteomicsAnalysisSkill failed:\n%s", traceback.format_exc())
            return OmicsAnalysisResult(
                omic_type=self.omic_type,
                top_biomarkers=[],
                n_significant=0,
                excel_path=None,
                qc_summary={},
                error=str(exc),
            )

    # ── Internal pipeline ─────────────────────────────────────────────────────

    def _run(
        self,
        data_path, sample_columns, group1_samples, group2_samples,
        group1_label, group2_label, analysis_mode, data_type,
        adj_pval_cutoff, log2fc_cutoff, missing_threshold,
        top_n, output_dir, file_name,
    ) -> Dict[str, Any]:

        # 1. Load
        df_raw = pd.read_csv(data_path, index_col=0)
        avail = [c for c in sample_columns if c in df_raw.columns]
        if not avail:
            raise ValueError(
                "None of the specified sample_columns are present in the data file.\n"
                f"Available columns: {list(df_raw.columns[:10])}"
            )
        data = df_raw[avail].apply(pd.to_numeric, errors="coerce")

        # 2. QC
        data_qc, qc_summary = self._qc(data, missing_threshold, data_type)

        # 3. Analysis
        if (
            analysis_mode == "supervised"
            and group1_samples
            and group2_samples
        ):
            results_df = self._supervised(
                data_qc, group1_samples, group2_samples,
                group1_label, group2_label,
            )
            sig_mask = (
                (results_df["adj_p_value"] < adj_pval_cutoff)
                & (results_df["log2_fold_change"].abs() >= log2fc_cutoff)
            )
            n_sig = int(sig_mask.sum())
        else:
            results_df = self._unsupervised(data_qc)
            n_sig = min(top_n, len(results_df))

        top_df = results_df.head(top_n).copy()
        top_biomarkers = top_df.to_dict("records")

        # 4. Excel export
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = Path(file_name).stem.replace(" ", "_")
        excel_path = str(Path(output_dir) / f"biomarkers_{safe_name}_{ts}.xlsx")

        self._export_excel(
            results_df, top_df, qc_summary, excel_path,
            group1_label, group2_label, analysis_mode,
            adj_pval_cutoff, log2fc_cutoff, file_name,
        )

        return {
            "omic_type":     self.omic_type,
            "top_biomarkers": top_biomarkers,
            "n_significant": n_sig,
            "excel_path":    excel_path,
            "qc_summary":    qc_summary,
            "error":         None,
        }

    # ── QC ────────────────────────────────────────────────────────────────────

    def _qc(
        self,
        data: pd.DataFrame,
        missing_threshold: float,
        data_type: str,
    ) -> Tuple[pd.DataFrame, Dict]:
        orig_proteins = len(data)
        orig_samples  = len(data.columns)

        # Log2 transform if values look like raw intensities
        log_done = False
        col_max = data.max().max()
        if pd.notna(col_max) and float(col_max) > 100:
            data = np.log2(data.replace(0, np.nan) + 1)
            log_done = True

        # Filter proteins with too many missing values across ALL samples
        miss_prot = data.isna().mean(axis=1)
        data = data.loc[miss_prot <= missing_threshold]

        # Filter samples with >80 % missing values
        miss_samp = data.isna().mean(axis=0)
        data = data.loc[:, miss_samp <= 0.80]

        # Half-minimum imputation (per protein)
        # Uses numpy to avoid ambiguous Series/DataFrame from duplicate index labels.
        arr = data.to_numpy(dtype=float, copy=True)
        nan_mask = np.isnan(arr)
        if nan_mask.any():
            with np.errstate(all="ignore"):
                row_half_min = np.nanmin(arr, axis=1, keepdims=True) / 2.0
            row_half_min = np.where(np.isnan(row_half_min), 0.0, row_half_min)
            arr[nan_mask] = np.broadcast_to(row_half_min, arr.shape)[nan_mask]
            data = pd.DataFrame(arr, index=data.index, columns=data.columns)

        qc_summary = {
            "proteins_input":   orig_proteins,
            "proteins_after_qc": len(data),
            "proteins_removed": orig_proteins - len(data),
            "samples_input":    orig_samples,
            "samples_after_qc": len(data.columns),
            "samples_removed":  orig_samples - len(data.columns),
            "log2_transformed": log_done,
            "missing_threshold": missing_threshold,
            "imputation_method": "half-minimum per protein",
        }
        return data, qc_summary

    # ── Supervised (two-group t-test) ─────────────────────────────────────────

    def _supervised(
        self,
        data: pd.DataFrame,
        g1_cols: List[str],
        g2_cols: List[str],
        g1_label: str,
        g2_label: str,
    ) -> pd.DataFrame:
        g1 = [c for c in g1_cols if c in data.columns]
        g2 = [c for c in g2_cols if c in data.columns]

        if not g1 or not g2:
            raise ValueError(
                f"Group columns not found in QC-filtered data.\n"
                f"Group1 requested: {g1_cols}\n"
                f"Group2 requested: {g2_cols}\n"
                f"Available columns: {list(data.columns)}"
            )

        if len(g1) < 2 or len(g2) < 2:
            raise ValueError(
                f"Differential analysis requires at least 2 samples per group. "
                f"Got: {g1_label}={len(g1)} sample(s), {g2_label}={len(g2)} sample(s). "
                f"Please assign more samples to each group."
            )

        rows = []
        for protein in data.index:
            v1 = data.loc[protein, g1].dropna().values.astype(float)
            v2 = data.loc[protein, g2].dropna().values.astype(float)

            if len(v1) < 2 or len(v2) < 2:
                continue

            m1, m2 = v1.mean(), v2.mean()

            try:
                _, pval = stats.ttest_ind(v1, v2, equal_var=False)
            except Exception:
                continue

            if np.isnan(pval):
                continue

            # log2 FC: data is already log2-transformed by QC if needed,
            # so difference = log2(G1/G2)
            if m2 == 0:
                log2fc = np.inf if m1 > 0 else 0.0
            else:
                log2fc = float(m1 - m2)

            rows.append({
                "protein":            protein,
                f"mean_{g1_label}":   round(float(m1), 4),
                f"mean_{g2_label}":   round(float(m2), 4),
                "log2_fold_change":   round(log2fc, 4),
                "p_value":            float(pval),
                f"n_{g1_label}":      len(v1),
                f"n_{g2_label}":      len(v2),
            })

        if not rows:
            raise ValueError("No proteins had enough values in both groups for t-test.")

        df = pd.DataFrame(rows)

        # BH FDR correction
        _, adj_pvals, _, _ = multipletests(df["p_value"].values, method="fdr_bh")
        df["adj_p_value"] = adj_pvals

        # Significance labels
        df["significance"] = "NS"
        hi  = (df["adj_p_value"] < 0.01)  & (df["log2_fold_change"].abs() >= 1.0)
        sig = (df["adj_p_value"] < 0.05)  & (df["log2_fold_change"].abs() >= 1.0)
        trn = (df["adj_p_value"] < 0.10)  & ~sig
        df.loc[trn,  "significance"] = "Trend"
        df.loc[sig,  "significance"] = "Significant"
        df.loc[hi,   "significance"] = "Highly Significant"

        df = df.sort_values("adj_p_value").reset_index(drop=True)
        df.insert(0, "rank", range(1, len(df) + 1))
        return df

    # ── Unsupervised (CV ranking) ─────────────────────────────────────────────

    def _unsupervised(self, data: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for protein in data.index:
            vals = data.loc[protein].dropna().values.astype(float)
            if len(vals) < 3:
                continue
            m = vals.mean()
            cv = (vals.std() / abs(m) * 100) if m != 0 else 0.0
            mad = float(np.median(np.abs(vals - np.median(vals))))
            rows.append({
                "protein":             protein,
                "mean_expression":     round(float(m), 4),
                "std_expression":      round(float(vals.std()), 4),
                "cv_percent":          round(cv, 2),
                "median_abs_deviation": round(mad, 4),
                "n_samples":           len(vals),
                "significance":        "Top Variable",
            })

        df = pd.DataFrame(rows)
        df = df.sort_values("cv_percent", ascending=False).reset_index(drop=True)
        df.insert(0, "rank", range(1, len(df) + 1))
        return df

    # ── Excel export ──────────────────────────────────────────────────────────

    def _export_excel(
        self,
        all_results: pd.DataFrame,
        top_results: pd.DataFrame,
        qc_summary: Dict,
        output_path: str,
        g1_label: str,
        g2_label: str,
        analysis_mode: str,
        adj_pval_cutoff: float,
        log2fc_cutoff: float,
        file_name: str,
    ) -> None:
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            top_results.to_excel(writer, sheet_name="Top Biomarkers", index=False)
            all_results.to_excel(writer, sheet_name="All Results",    index=False)

            qc_df = pd.DataFrame(
                [{"Parameter": k.replace("_", " ").title(), "Value": v}
                 for k, v in qc_summary.items()]
            )
            qc_df.to_excel(writer, sheet_name="QC Summary", index=False)

            params = {
                "Analysis Date":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "File":             file_name,
                "Mode":             analysis_mode.capitalize(),
                "Group 1":          g1_label if analysis_mode == "supervised" else "N/A",
                "Group 2":          g2_label if analysis_mode == "supervised" else "N/A",
                "Adj P-value Cut":  adj_pval_cutoff,
                "Log2 FC Cut":      log2fc_cutoff,
                "Statistical Test": ("Welch t-test" if analysis_mode == "supervised"
                                     else "CV ranking"),
                "FDR Method":       ("Benjamini-Hochberg" if analysis_mode == "supervised"
                                     else "N/A"),
                "Imputation":       "Half-minimum per protein",
            }
            params_df = pd.DataFrame(
                [{"Parameter": k, "Value": v} for k, v in params.items()]
            )
            params_df.to_excel(writer, sheet_name="Parameters", index=False)

        self._format_workbook(output_path, analysis_mode)

    def _format_workbook(self, path: str, analysis_mode: str) -> None:
        wb = openpyxl.load_workbook(path)
        hdr_fill, hdr_font = _header_style()
        border = _thin_border()

        sig_fills = {
            "Highly Significant": PatternFill(start_color=_HIGHLY_SIG,  end_color=_HIGHLY_SIG,  fill_type="solid"),
            "Significant":        PatternFill(start_color=_SIGNIFICANT, end_color=_SIGNIFICANT, fill_type="solid"),
            "Trend":              PatternFill(start_color=_TREND,       end_color=_TREND,       fill_type="solid"),
            "Top Variable":       PatternFill(start_color=_VARIABLE,    end_color=_VARIABLE,    fill_type="solid"),
        }

        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]

            # Header row styling
            for cell in ws[1]:
                cell.fill = hdr_fill
                cell.font = hdr_font
                cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
                cell.border = border
            ws.row_dimensions[1].height = 28

            # Auto-fit column widths
            for col in ws.columns:
                letter = get_column_letter(col[0].column)
                max_len = max(
                    (len(str(c.value)) for c in col if c.value is not None),
                    default=8,
                )
                ws.column_dimensions[letter].width = min(max_len + 4, 40)

            # Row colour by significance (Top Biomarkers sheet only)
            if sheet_name == "Top Biomarkers":
                sig_col_idx = next(
                    (i for i, c in enumerate(ws[1], 1)
                     if c.value and "significance" in str(c.value).lower()),
                    None,
                )
                for row in ws.iter_rows(min_row=2):
                    sig_val = (str(row[sig_col_idx - 1].value)
                               if sig_col_idx and row[sig_col_idx - 1].value
                               else "NS")
                    fill = sig_fills.get(sig_val)
                    for cell in row:
                        if fill:
                            cell.fill = fill
                        cell.border = border
                        cell.alignment = Alignment(horizontal="center")

            ws.freeze_panes = "A2"

        # Legend sheet
        legend_ws = wb.create_sheet("Legend")
        legend_data = [
            ("Colour",       "Significance",        "Criteria"),
            ("Dark Green",   "Highly Significant",  "Adj p < 0.01 AND |Log2FC| ≥ 1.0"),
            ("Light Green",  "Significant",         "Adj p < 0.05 AND |Log2FC| ≥ 1.0"),
            ("Yellow",       "Trend",               "Adj p < 0.10"),
            ("Pale Blue",    "Top Variable",        "Top CV proteins (unsupervised)"),
            ("White",        "Not Significant (NS)", "Does not meet above thresholds"),
        ]
        colour_fills = [None, sig_fills["Highly Significant"], sig_fills["Significant"],
                        sig_fills["Trend"], sig_fills["Top Variable"], None]

        for r_idx, (row_data, row_fill) in enumerate(zip(legend_data, colour_fills), 1):
            for c_idx, val in enumerate(row_data, 1):
                cell = legend_ws.cell(row=r_idx, column=c_idx, value=val)
                cell.border = border
                if r_idx == 1:
                    cell.fill = hdr_fill
                    cell.font = hdr_font
                elif row_fill:
                    cell.fill = row_fill
                cell.alignment = Alignment(horizontal="left")
        for col in legend_ws.columns:
            letter = get_column_letter(col[0].column)
            legend_ws.column_dimensions[letter].width = 40

        wb.save(path)
