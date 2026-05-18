"""
skills — omic analysis skills for the biomarker discovery platform.

Public API
----------
BaseOmicsSkill          Abstract base; subclass this to add a new omic type.
OmicsAnalysisResult     Standardised TypedDict returned by every skill.
OmicsSkillRegistry      Thread-safe registry that maps omic_type → skill instance.
ProteomicsAnalysisSkill Python-side proteomics analysis (t-test / limma / CV).
RAnalysisSkill          R-side proteomics analysis (limma via Rscript subprocess).
DualEngineSkill         Combines Python + R into an intersected top-biomarker list.
"""
from skills.base_skill import BaseOmicsSkill, OmicsAnalysisResult
from skills.omics_registry import OmicsSkillRegistry
from skills.proteomics_analysis import ProteomicsAnalysisSkill

__all__ = [
    "BaseOmicsSkill",
    "OmicsAnalysisResult",
    "OmicsSkillRegistry",
    "ProteomicsAnalysisSkill",
]
