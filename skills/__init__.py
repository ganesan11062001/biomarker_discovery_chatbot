"""
skills — omic analysis skills for the biomarker discovery platform.

Public API
----------
BaseOmicsSkill          Abstract base; subclass this to add a new omic type.
OmicsAnalysisResult     Standardised TypedDict returned by every skill.
OmicsSkillRegistry      Thread-safe registry that maps omic_type → skill instance.
ProteomicsAnalysisSkill Ready-to-use proteomics skill (t-test / CV ranking).
PooledFoldChangeSkill   Pooled design skill (n=1 per group, fold-change only).
"""
from skills.base_skill import BaseOmicsSkill, OmicsAnalysisResult
from skills.omics_registry import OmicsSkillRegistry
from skills.pooled_fold_change import PooledFoldChangeSkill
from skills.proteomics_analysis import ProteomicsAnalysisSkill

__all__ = [
    "BaseOmicsSkill",
    "OmicsAnalysisResult",
    "OmicsSkillRegistry",
    "PooledFoldChangeSkill",
    "ProteomicsAnalysisSkill",
]
