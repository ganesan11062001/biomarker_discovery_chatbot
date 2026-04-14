"""
skills/omics_registry.py

Thread-safe registry that maps omic-type identifiers to their analysis skills.

Usage
-----
The registry is populated once at application startup inside BiomarkerAgent.
New omic types are added by registering a concrete BaseOmicsSkill subclass —
no changes to BiomarkerAgent or the API layer are required.

    registry = OmicsSkillRegistry()
    registry.register(ProteomicsAnalysisSkill())
    # Future additions:
    # registry.register(TranscriptomicsSkill())
    # registry.register(MetabolomicsSkill())

    skill = registry.get("proteomics")   # → ProteomicsAnalysisSkill instance
    skill.execute(**params)
"""
from __future__ import annotations

import logging
import threading
from typing import Dict, List

from skills.base_skill import BaseOmicsSkill

logger = logging.getLogger(__name__)


class OmicsSkillRegistry:
    """
    Thread-safe registry for omic analysis skills.

    One instance lives inside BiomarkerAgent; skills are registered during
    __init__ and only looked up (never mutated) at request time.
    """

    def __init__(self) -> None:
        self._lock: threading.Lock = threading.Lock()
        self._skills: Dict[str, BaseOmicsSkill] = {}

    # ── Registration ──────────────────────────────────────────────────────────

    def register(self, skill: BaseOmicsSkill) -> None:
        """
        Register a skill instance under its omic_type key.

        Re-registering the same omic_type replaces the previous skill
        (useful for testing / hot-swap scenarios).

        Parameters
        ----------
        skill:
            A concrete BaseOmicsSkill instance.

        Raises
        ------
        TypeError
            If `skill` is not a BaseOmicsSkill subclass.
        """
        if not isinstance(skill, BaseOmicsSkill):
            raise TypeError(
                f"Expected a BaseOmicsSkill instance, got {type(skill).__name__!r}."
            )
        with self._lock:
            if skill.omic_type in self._skills:
                logger.warning(
                    "Replacing existing skill for omic_type=%r (%s → %s)",
                    skill.omic_type,
                    type(self._skills[skill.omic_type]).__name__,
                    type(skill).__name__,
                )
            self._skills[skill.omic_type] = skill
            logger.debug("Registered skill %r for omic_type=%r", type(skill).__name__, skill.omic_type)

    # ── Lookup ────────────────────────────────────────────────────────────────

    def get(self, omic_type: str) -> BaseOmicsSkill:
        """
        Return the skill registered for `omic_type`.

        Parameters
        ----------
        omic_type:
            Omic-type identifier (e.g. "proteomics", "transcriptomics").

        Raises
        ------
        KeyError
            If no skill is registered for the requested omic type, with a
            helpful message listing what is currently available.
        """
        with self._lock:
            skill = self._skills.get(omic_type)
        if skill is None:
            available = self.available()
            raise KeyError(
                f"No skill registered for omic_type={omic_type!r}. "
                f"Available: {available}. "
                "Register a BaseOmicsSkill subclass to add support."
            )
        return skill

    def get_or_default(self, omic_type: str, default: str = "proteomics") -> BaseOmicsSkill:
        """
        Return the skill for `omic_type`, falling back to `default` if not found.

        Useful when the omic type is inferred (e.g. from file content) and
        may not always be set explicitly in state.
        """
        try:
            return self.get(omic_type)
        except KeyError:
            logger.warning(
                "omic_type=%r not registered; falling back to %r.", omic_type, default
            )
            return self.get(default)

    # ── Introspection ─────────────────────────────────────────────────────────

    def available(self) -> List[str]:
        """Return the sorted list of registered omic-type identifiers."""
        with self._lock:
            return sorted(self._skills.keys())

    def __contains__(self, omic_type: str) -> bool:
        with self._lock:
            return omic_type in self._skills

    def __repr__(self) -> str:
        return f"<OmicsSkillRegistry skills={self.available()}>"
