"""
core/action_units.py
GenoMAS-style action unit loader.

Action units are JSON-defined named tasks describing what each agent does at
each step. They serve three purposes:
  1. Documentation — the canonical, readable definition of the pipeline.
  2. Customisability — change behaviour without editing Python (edit JSON).
  3. Prompt augmentation — agents can paste relevant units into LLM context
     so the model knows exactly what step it's executing and what's expected.

Loading is lazy + cached. Files live under prompts/action_units/.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_ACTION_UNITS_DIR = Path(__file__).parent.parent / "prompts" / "action_units"


@dataclass
class ActionUnit:
    """A named step in an agent's pipeline."""
    name:          str
    instruction:   str
    parameters:    List[str]   = field(default_factory=list)
    fail_severity: str         = "minor"
    code_template: Optional[str] = None


@dataclass
class ActionUnitSet:
    """All action units for one agent (e.g. biomarker, ingestion)."""
    units: List[ActionUnit]

    def by_name(self, name: str) -> Optional[ActionUnit]:
        for u in self.units:
            if u.name == name:
                return u
        return None

    def names(self) -> List[str]:
        return [u.name for u in self.units]

    def as_prompt_block(self, only_names: Optional[List[str]] = None) -> str:
        """Render units as a markdown block suitable for system-prompt injection."""
        selected = self.units
        if only_names:
            selected = [u for u in self.units if u.name in only_names]
        lines = []
        for u in selected:
            params = f"  parameters: {u.parameters}\n" if u.parameters else ""
            lines.append(
                f"- **{u.name}** ({u.fail_severity})\n"
                f"  {u.instruction}\n"
                f"{params}"
            )
        return "\n".join(lines)


def _load_file(path: Path) -> ActionUnitSet:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        raw_units = data.get("units", [])
        return ActionUnitSet(units=[
            ActionUnit(
                name=str(u.get("name", "")),
                instruction=str(u.get("instruction", "")),
                parameters=list(u.get("parameters") or []),
                fail_severity=str(u.get("fail_severity", "minor")),
                code_template=u.get("code_template"),
            )
            for u in raw_units
            if u.get("name")
        ])
    except Exception as exc:
        logger.warning("Failed to load action units from %s: %s", path, exc)
        return ActionUnitSet(units=[])


@lru_cache(maxsize=8)
def load_action_units(agent: str) -> ActionUnitSet:
    """
    Load the action unit set for an agent by name.

    `agent` is the file stem under prompts/action_units/:
      'biomarker'   → biomarker_action_units.json
      'ingestion'   → ingestion_action_units.json
      'query_data'  → query_data_action_units.json
    """
    path = _ACTION_UNITS_DIR / f"{agent}_action_units.json"
    if not path.exists():
        logger.debug("No action units file for agent %r at %s", agent, path)
        return ActionUnitSet(units=[])
    return _load_file(path)
