"""
skills/base_skill.py

Abstract base class for all omic-type analysis skills.

Design
------
Adding a new omic type (e.g. transcriptomics, metabolomics) requires only:
  1. Subclass BaseOmicsSkill
  2. Set the `omic_type` property
  3. Implement `execute(**kwargs) -> OmicsAnalysisResult`
  4. Register the skill in agents/biomarker_agent.py

The OmicsAnalysisResult TypedDict defines the contract that every skill
must honour so the BiomarkerAgent can handle any omic type uniformly.

Supported omic types (current and planned)
-------------------------------------------
  proteomics      ← implemented (ProteomicsAnalysisSkill)
  transcriptomics ← planned (RNA-seq / microarray)
  metabolomics    ← planned (LC-MS metabolite intensities)
  lipidomics      ← planned (lipid species intensities)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypedDict


# ── Standard result contract ──────────────────────────────────────────────────

class OmicsAnalysisResult(TypedDict, total=False):
    """
    Standardised output returned by every BaseOmicsSkill.execute() call.

    Fields marked Optional are populated only when applicable
    (e.g. log2_fold_change is absent for unsupervised CV analyses).
    """
    omic_type:      str                     # mirrors the skill's omic_type
    top_biomarkers: List[Dict[str, Any]]    # ranked list of biomarker dicts
    n_significant:  int                     # proteins/features meeting cutoffs
    excel_path:     Optional[str]           # absolute path to Excel report
    qc_summary:     Dict[str, Any]          # QC metrics (proteins removed, etc.)
    error:          Optional[str]           # non-None on failure
    analysis_code:  Optional[str]           # self-contained reproducible Python script


# ── Abstract base ─────────────────────────────────────────────────────────────

class BaseOmicsSkill(ABC):
    """
    Abstract base for all single-omic analysis skills.

    Every concrete skill must declare its `omic_type` and implement
    `execute(**kwargs)`.  The keyword arguments accepted by execute()
    are skill-specific; the return value must conform to OmicsAnalysisResult.

    Example
    -------
    >>> class TranscriptomicsSkill(BaseOmicsSkill):
    ...     @property
    ...     def omic_type(self) -> str:
    ...         return "transcriptomics"
    ...
    ...     def execute(self, **kwargs) -> OmicsAnalysisResult:
    ...         # load counts, run DESeq2-equivalent, export Excel …
    ...         return OmicsAnalysisResult(
    ...             omic_type="transcriptomics",
    ...             top_biomarkers=[...],
    ...             n_significant=42,
    ...             excel_path="/outputs/deseq2_results.xlsx",
    ...             qc_summary={...},
    ...             error=None,
    ...         )
    """

    @property
    @abstractmethod
    def omic_type(self) -> str:
        """
        Unique identifier for the omic layer this skill analyses.

        Convention: lowercase singular noun — "proteomics", "transcriptomics",
        "metabolomics", "lipidomics", "epigenomics", etc.
        """

    @abstractmethod
    def execute(self, **kwargs: Any) -> OmicsAnalysisResult:
        """
        Run the full analysis pipeline for this omic type.

        Parameters
        ----------
        **kwargs:
            Skill-specific parameters.  Typical keys (all optional):
            - data_path         : str  — path to the intensity matrix file
            - sample_columns    : list — numeric sample column names
            - group1_samples    : list — sample columns for group 1
            - group2_samples    : list — sample columns for group 2
            - group1_label      : str  — human-readable label for group 1
            - group2_label      : str  — human-readable label for group 2
            - analysis_mode     : str  — "supervised" | "unsupervised"
            - data_type         : str  — instrument/platform hint
            - adj_pval_cutoff   : float
            - log2fc_cutoff     : float
            - missing_threshold : float
            - top_n             : int
            - output_dir        : str
            - file_name         : str

        Returns
        -------
        OmicsAnalysisResult
            Standardised result dict.  Set ``error`` to a non-empty string
            on failure; all other fields are optional in that case.
        """

    def __repr__(self) -> str:
        return f"<{type(self).__name__} omic_type={self.omic_type!r}>"
